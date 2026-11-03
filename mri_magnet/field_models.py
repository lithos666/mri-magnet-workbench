"""One immutable physical model per optimization run, with bounded preparation.

Geometry identity and evaluation identity have different purposes. A model
fingerprint includes its material-response discretization; an evaluation also
includes the exact observer points, integration weights and metric version.
"""
from __future__ import annotations

from collections import OrderedDict
from copy import deepcopy
import hashlib
import json
import math
import time

import numpy as np

from .field import FieldEngine
from .identity import canonical_physical_sha256
from .material_response import CoupledResponseEngine, preflight_response, _options
from .serialization import jsonable


MODEL_VERSION = 'v6-linear-recoil-cells-2-exact-tensor'
PHYSICS_DEFAULTS = {
    'field_model':'coupled', 'quadrature_order':2, 'cells_per_axis':1,
    'residual_tolerance':1e-10, 'prepared_cache_size':4, 'max_unknowns':1200,
    'max_matrix_mb':128., 'max_magnets':10000, 'chunk_points':48,
    'symmetry_reduction':True, 'interaction_backend':'analytic_tensor',
}


def physics_options(config):
    raw = config.get('physics')
    if raw is None:
        return dict(PHYSICS_DEFAULTS, field_model='legacy')
    if not isinstance(raw, dict): raise ValueError('physics must be an object')
    unknown = set(raw)-set(PHYSICS_DEFAULTS)
    if unknown: raise ValueError('Unknown physics settings: '+', '.join(sorted(unknown)))
    result = dict(PHYSICS_DEFAULTS, **raw)
    if result['field_model'] not in ('coupled','legacy'):
        raise ValueError('physics.field_model must be coupled or legacy')
    capacity = result['prepared_cache_size']
    if isinstance(capacity, bool) or not math.isfinite(float(capacity)) or int(capacity)!=capacity or capacity<1:
        raise ValueError('physics.prepared_cache_size must be a positive integer')
    result['prepared_cache_size'] = int(capacity)
    _options({k:v for k,v in result.items() if k not in ('field_model','prepared_cache_size')})
    return result


def response_options(config):
    return {key:value for key,value in physics_options(config).items()
            if key not in ('field_model','prepared_cache_size')}


def model_spec(config):
    options = physics_options(config)
    if options['field_model']=='legacy':
        model = config.get('polarization_model','remanence')
        if model not in ('remanence','isolated_cube_self_demag'):
            raise ValueError('Unknown legacy polarization_model')
        return {'version':MODEL_VERSION, 'field_model':'legacy', 'polarization_model':model}
    return {'version':MODEL_VERSION, 'field_model':'coupled', 'polarization_model':'coupled_uniform_cells',
            **{key:options[key] for key in ('quadrature_order','cells_per_axis','residual_tolerance','symmetry_reduction','interaction_backend')}}


def _digest(value):
    return hashlib.sha256(json.dumps(jsonable(value),sort_keys=True,separators=(',',':'),
                                    allow_nan=False).encode('utf-8')).hexdigest()


def model_fingerprint(config):
    return _digest(model_spec(config))


def evaluation_fingerprint(config, sample, metric_version='field_metrics_v1'):
    points=np.asarray(sample['points_mm'],dtype=float)
    weights=np.asarray(sample.get('weights',np.ones(len(points))),dtype=float)
    if points.ndim!=2 or points.shape[1]!=3 or not np.all(np.isfinite(points)):
        raise ValueError('Evaluation fingerprint requires finite Nx3 points')
    if weights.shape!=(len(points),) or not np.all(np.isfinite(weights)) or np.any(weights<0) or weights.sum()<=0:
        raise ValueError('Evaluation fingerprint requires matching nonnegative weights')
    return _digest({'model':model_spec(config),'points_mm':points,'weights':weights,
                    'target':config.get('target'), 'metric_version':metric_version,
                    'density_kg_m3':config.get('density_kg_m3',7500.),
                    'sample_metadata':{key:value for key,value in sample.items()
                                       if key not in ('points_mm','weights')}})


class CoupledFieldEngine:
    """Per-process cache of self-consistent responses, never frozen across moves."""
    polarization_model = 'coupled_uniform_cells'

    def __init__(self, config, cancel_check=None, status_callback=None):
        self.config=deepcopy(config)
        self.options=physics_options(config)
        self.settings=response_options(config)
        self.model_fingerprint=model_fingerprint(config)
        self.model_spec=model_spec(config)
        self.cancel_check=cancel_check
        self.status_callback=status_callback
        self._prepared=OrderedDict()
        self.diagnostics=None
        self.ring_summary=[]
        self.reset_statistics()

    def reset_statistics(self):
        self.statistics={'engine_calls':0,'field_points':0,'field_seconds':0.,
                         'preparations':0,'preparation_seconds':0.,'prepared_cache_hits':0,
                         'preparation_failures':0,'imported_preparations':0,
                         'matrix_builds':0,'self_consistent_solves':0,'peak_prepared_entries':0,
                         'field_tensor_fallback_rows':0}

    def preflight(self, design):
        return preflight_response(design,self.settings)

    def _key(self, design):
        return (canonical_physical_sha256(design,self.polarization_model),self.model_fingerprint)

    def _remember(self,key,prepared):
        self._prepared[key]=prepared
        self._prepared.move_to_end(key)
        while len(self._prepared)>self.options['prepared_cache_size']:
            self._prepared.popitem(last=False)
        self.statistics['peak_prepared_entries']=max(self.statistics['peak_prepared_entries'],len(self._prepared))
        self.diagnostics=prepared.diagnostics
        self.ring_summary=prepared.ring_summary
        return prepared

    def prepare(self, design):
        if self.cancel_check is not None: self.cancel_check()
        key=self._key(design)
        if key in self._prepared:
            self.statistics['prepared_cache_hits']+=1
            return self._remember(key,self._prepared[key])
        start=time.perf_counter()
        try:
            prepared=CoupledResponseEngine(design,self.settings,self.cancel_check,self.status_callback)
        except BaseException:
            self.statistics['preparation_failures']+=1
            self.statistics['preparation_seconds']+=time.perf_counter()-start
            raise
        self.statistics['preparation_seconds']+=time.perf_counter()-start
        self.statistics['preparations']+=1
        if design.reference_mur!=1.:
            self.statistics['matrix_builds']+=1
            self.statistics['self_consistent_solves']+=1
        return self._remember(key,prepared)

    def field(self, design, points_mm):
        prepared=self.prepare(design)
        if self.cancel_check is not None:self.cancel_check()
        started=time.perf_counter()
        prior_fallback=prepared.statistics.get('field_tensor_fallback_rows',0)
        values=prepared.field(design,points_mm)
        self.statistics['field_tensor_fallback_rows']+=prepared.statistics.get('field_tensor_fallback_rows',0)-prior_fallback
        self.statistics['field_seconds']+=time.perf_counter()-started
        self.statistics['engine_calls']+=1
        self.statistics['field_points']+=len(values)
        return values

    def export_prepared(self,design):
        """Optional ephemeral IPC payload for the monitor, not a public file."""
        key=self._key(design)
        if key not in self._prepared:return None
        return {'model_fingerprint':self.model_fingerprint,
                'response':self._prepared[key].export_prepared()}

    def import_prepared(self,design,payload):
        if not isinstance(payload,dict) or payload.get('model_fingerprint')!=self.model_fingerprint:
            raise ValueError('Prepared response uses a different field model')
        key=self._key(design)
        if key in self._prepared:return self._remember(key,self._prepared[key])
        response=CoupledResponseEngine(design,self.settings,prepared_response=payload['response'])
        self.statistics['imported_preparations']+=1
        return self._remember(key,response)


def make_field_engine(config, cancel_check=None, status_callback=None):
    options=physics_options(config)
    if options['field_model']=='coupled':
        return CoupledFieldEngine(config,cancel_check,status_callback)
    engine=FieldEngine(config.get('chunk_points',96),config.get('polarization_model','remanence'),
                       prepared_cache_size=options['prepared_cache_size'])
    engine.model_fingerprint=model_fingerprint(config)
    engine.model_spec=model_spec(config)
    return engine
