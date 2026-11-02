"""Separate weighted integration statistics from extrema, retaining vector diagnostics."""
from __future__ import annotations
import numpy as np

def field_metrics(B_T,weights=None,center_B_T=None):
    B=np.asarray(B_T,dtype=float)
    if B.ndim!=2 or B.shape[1]!=3 or not len(B) or not np.all(np.isfinite(B)):raise ValueError('Expected finite Nx3 field')
    w=np.ones(len(B))/len(B) if weights is None else np.asarray(weights,dtype=float)
    if w.shape!=(len(B),) or not np.all(np.isfinite(w)) or np.any(w<0) or w.sum()<=0:raise ValueError('Invalid integration weights')
    w=w/w.sum();norm=np.linalg.norm(B,axis=1);mean=float(w@norm);mean_z=float(w@B[:,2])
    if mean<1e-12:raise ValueError('Mean field too close to zero to define relative homogeneity')
    pp=float(np.ptp(norm));rms=float(np.sqrt(w@((norm-mean)**2)))
    center=np.asarray(B[0] if center_B_T is None else center_B_T)
    return {"mean_norm_T":mean,"min_norm_T":float(norm.min()),"max_norm_T":float(norm.max()),
            "delta_norm_T":pp,"pp_norm_ppm":pp/mean*1e6,"rms_norm_ppm":rms/mean*1e6,
            "mean_Bz_T":mean_z,"pp_Bz_ppm":float(np.ptp(B[:,2])/abs(mean_z)*1e6) if abs(mean_z)>1e-12 else None,
            "center_Bz_T":float(center[2]),"center_norm_T":float(np.linalg.norm(center)),
            "center_Bx_T":float(center[0]),"center_By_T":float(center[1]),
            "max_transverse_T":float(np.linalg.norm(B[:,:2],axis=1).max()),"n_points":len(B)}

def evaluate_design(design,engine,sample,density_kg_m3=7500.):
    B=engine.field(design,sample['points_mm'])
    center_point=np.array([0.,0.,float(sample.get('z_mm',0.))])
    center=B[0] if np.allclose(sample['points_mm'][0],center_point) else engine.field(design,[center_point])[0]
    metrics=field_metrics(B,sample.get('weights'),center)
    metrics.update({"pointset":sample['name'],"magnet_count":design.magnet_count,
        "mass_magnet_kg":design.magnet_count*(design.cube_mm*1e-3)**3*density_kg_m3,
        "mass_density_assumption_kg_m3":density_kg_m3,"polarization_model":engine.polarization_model,
        "integration_measure":sample.get('measure','volume'),"evaluation_geometry":sample.get('geometry','sphere'),
        "evaluation_z_mm":sample.get('z_mm'),"evaluation_diameter_mm":sample.get('diameter_mm')})
    return metrics,B
