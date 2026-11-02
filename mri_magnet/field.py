"""Magpylib 5 SI analytical cuboid fields, summed with bounded memory."""
from __future__ import annotations
from collections import OrderedDict
import time
import numpy as np
import magpylib as magpy
from scipy.spatial.transform import Rotation
from .geometry import Design, RingSpec
from .identity import canonical_physical_data, canonical_physical_sha256

class FieldEngine:
    def __init__(self, chunk_points=96, polarization_model="remanence", prepared_cache_size=8):
        self.chunk_points=int(chunk_points)
        self.polarization_model=polarization_model
        if self.chunk_points<1:raise ValueError('chunk_points must be positive')
        if polarization_model not in ('remanence','isolated_cube_self_demag'):
            raise ValueError('Unknown polarization model')
        if int(prepared_cache_size) != prepared_cache_size or prepared_cache_size < 1:
            raise ValueError('prepared_cache_size must be a positive integer')
        self.prepared_cache_size = int(prepared_cache_size)
        self._prepared = OrderedDict()
        self._design_keys = OrderedDict()
        self._last_design = None
        self._last_prepared = None
        self.reset_statistics()

    def reset_statistics(self):
        self.statistics = {'engine_calls': 0, 'field_points': 0, 'preparations': 0,
                           'prepared_cache_hits': 0, 'field_seconds': 0.0,
                           'preparation_seconds': 0.0}

    def _prepare(self, design):
        if self._last_design is design:
            self.statistics['prepared_cache_hits'] += 1
            return self._last_prepared
        key = self._design_keys.get(design)
        if key is None:
            key = canonical_physical_sha256(design, self.polarization_model)
            self._design_keys[design] = key
            while len(self._design_keys) > self.prepared_cache_size * 2:
                self._design_keys.popitem(last=False)
        if key in self._prepared:
            prepared = self._prepared.pop(key)
            self._prepared[key] = prepared
            self.statistics['prepared_cache_hits'] += 1
        else:
            start = time.perf_counter()
            data = canonical_physical_data(design, self.polarization_model)
            canonical = Design(tuple(RingSpec('physical_'+str(i), i, 0, **ring)
                                     for i, ring in enumerate(data['rings'])),
                               cube_mm=data['cube_mm'], remanence_T=data['remanence_T'],
                               reference_mur=data['reference_mur'])
            arr = canonical.magnet_arrays()
            pos = arr['positions_mm'] * 1e-3
            theta = np.deg2rad(arr['radial_angles_deg'])
            cs, sn = np.cos(theta), np.sin(theta)
            north = arr['north_vectors']
            local_n = np.column_stack((cs*north[:,0]+sn*north[:,1],
                                       -sn*north[:,0]+cs*north[:,1], north[:,2]))
            polarization = canonical.remanence_T
            if self.polarization_model == 'isolated_cube_self_demag':
                polarization /= 1 + (canonical.reference_mur-1)/3
            local_J = local_n * polarization
            for array in (pos, cs, sn, local_J):
                array.setflags(write=False)
            prepared = (pos, cs, sn, local_J, canonical.cube_mm*1e-3)
            self._prepared[key] = prepared
            while len(self._prepared) > self.prepared_cache_size:
                self._prepared.popitem(last=False)
            self.statistics['preparations'] += 1
            self.statistics['preparation_seconds'] += time.perf_counter()-start
        self._last_design, self._last_prepared = design, prepared
        return prepared

    def field(self,design,points_mm):
        started = time.perf_counter()
        pts=np.asarray(points_mm,dtype=float).reshape(-1,3)
        if not np.all(np.isfinite(pts)):raise ValueError('Observer coordinates must be finite')
        self.statistics['engine_calls'] += 1
        self.statistics['field_points'] += len(pts)
        pos, cs, sn, local_J, cube_m = self._prepare(design)
        n=len(pos);out=np.empty((len(pts),3))
        for start in range(0,len(pts),self.chunk_points):
            section=pts[start:start+self.chunk_points]*1e-3
            delta=section[:,None,:]-pos[None,:,:]
            local=np.empty_like(delta)
            local[:,:,0]=delta[:,:,0]*cs+delta[:,:,1]*sn
            local[:,:,1]=-delta[:,:,0]*sn+delta[:,:,1]*cs
            local[:,:,2]=delta[:,:,2]
            local_b=magpy.getB('Cuboid',observers=local.reshape(-1,3),
                dimension=(cube_m,)*3,
                polarization=np.tile(local_J,(len(section),1))).reshape(len(section),n,3)
            bx=local_b[:,:,0]*cs-local_b[:,:,1]*sn
            by=local_b[:,:,0]*sn+local_b[:,:,1]*cs
            out[start:start+len(section)]=np.column_stack((bx.sum(axis=1),by.sum(axis=1),local_b[:,:,2].sum(axis=1)))
        if not np.all(np.isfinite(out)):raise FloatingPointError('Non-finite analytical field')
        self.statistics['field_seconds'] += time.perf_counter()-started
        return out

    def object_reference(self,design,points_mm):
        """Independent public-object transformation path for small verification cases."""
        arr=design.magnet_arrays();sources=[]
        polarization=design.remanence_T
        if self.polarization_model=='isolated_cube_self_demag':
            polarization/=1+(design.reference_mur-1)/3
        for p,n,a in zip(arr['positions_mm'],arr['north_vectors'],arr['radial_angles_deg']):
            rotation=Rotation.from_euler('z',a,degrees=True)
            sources.append(magpy.magnet.Cuboid(position=p*1e-3,dimension=(design.cube_mm*1e-3,)*3,
                orientation=rotation,polarization=rotation.inv().apply(n*polarization)))
        return np.asarray(magpy.getB(sources,np.asarray(points_mm)*1e-3,sumup=True)).reshape(-1,3)
