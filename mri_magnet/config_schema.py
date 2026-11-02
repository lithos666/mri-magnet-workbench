"""One canonical schema, with explicit migration of historical flat inputs."""
from copy import deepcopy
from .workbench_config import default_config, merge_config, validate_spec

SCHEMA_VERSION=3


def _engineering_aliases(raw):
    """Accept engineering-document names without creating a second input authority."""
    raw=deepcopy(raw)
    materials=raw.pop('MaterialsAndManufacturing',None)
    if materials is not None:
        raw['manufacturing']=merge_config(materials,raw.get('manufacturing',{}))
    constraints=raw.pop('Constraints',None)
    if constraints is not None:
        raw['envelope']=merge_config(constraints.get('envelope',{}),raw.get('envelope',{}))
    envelope=raw.get('envelope',{})
    heights=[]
    if 'max_total_height_mm' in envelope: heights.append(float(envelope.pop('max_total_height_mm')))
    for key in ('max_half_height_mm','half_length_mm'):
        if key in envelope: heights.append(2*float(envelope.pop(key)))
    if heights:
        if 'max_length_mm' in envelope: heights.append(float(envelope['max_length_mm']))
        if max(heights)-min(heights)>1e-9: raise ValueError('轴向总高与半高配置冲突')
        envelope['max_length_mm']=heights[0]
    process=raw.get('manufacturing',{})
    for key in ('printer_bed_mm','machine_bed_mm'):
        bed=process.get(key)
        if isinstance(bed,dict):
            process[key]=[bed['bed_x_mm'],bed['bed_y_mm'],bed['bed_z_mm']]
            if key=='printer_bed_mm' and 'margin_mm' in bed:
                if 'bed_margin_per_side_mm' in process and float(process['bed_margin_per_side_mm'])!=float(bed['margin_mm']):
                    raise ValueError('打印床 margin_mm 与 bed_margin_per_side_mm 冲突')
                process['bed_margin_per_side_mm']=bed['margin_mm']
    return raw


def canonicalize_config(raw, base=None, validate=True):
    raw=deepcopy(raw or {})
    if 'canonical_config' in raw:raw=deepcopy(raw['canonical_config'])
    raw=_engineering_aliases(raw)
    result=deepcopy(base) if base is not None else default_config()
    explicit_target='target' in raw
    old_disk='fov_diameter_mm' in raw
    old_sphere='dsv_diameter_mm' in raw
    if old_disk and old_sphere:raise ValueError('配置同时指定 fov_diameter_mm 与 dsv_diameter_mm；请用唯一的 target.geometry 明确评价域')
    if not explicit_target and (old_disk or old_sphere):
        raw['target']={'geometry':'sphere' if old_sphere else 'disk',
            'diameter_mm':raw['dsv_diameter_mm' if old_sphere else 'fov_diameter_mm'],
            'z_mm':0. if old_sphere else raw.get('fov_z_mm',0.)}
        # A partial ROI edit must not turn a constraint-only project into a
        # source-loading project. Standalone historical files retain seeded.
        raw.setdefault('mode',result.get('mode','de_novo') if base is not None else 'seeded')
    elif explicit_target and (old_disk or old_sphere):
        geometry='sphere' if old_sphere else 'disk'
        value=raw['dsv_diameter_mm' if old_sphere else 'fov_diameter_mm']
        target=raw['target']
        if target.get('geometry',geometry)!=geometry or target.get('diameter_mm',value)!=value:
            raise ValueError('target 与旧式评价域参数冲突；请仅保留 target')
    if 'max_evaluations' in raw:raw.setdefault('solver',{})['max_evaluations']=raw['max_evaluations']
    if 'optimization_options' in raw:
        allowed=set(result.get('solver',{}))
        raw.setdefault('solver',{}).update({k:v for k,v in raw['optimization_options'].items() if k in allowed})
    if raw.get('source_overrides',{}).get('enabled'):
        raw['magnet']={k:raw['source_overrides'][k] for k in ('cube_mm','remanence_T','reference_mur') if k in raw['source_overrides']}
    req=raw.get('requirements',{})
    if req.get('min_mean_norm_mT') is not None:raw.setdefault('target',{})['min_field_mT']=req['min_mean_norm_mT']
    if req.get('max_pp_norm_ppm') is not None:raw.setdefault('target',{})['max_ppm']=req['max_pp_norm_ppm']
    sampling=raw.get('sampling',{})
    for old,new in (('training_area','training_volume'),('training_boundary','training_surface'),
                    ('validation_area','validation_volume'),('validation_boundary','validation_surface')):
        old_key='quick_'+old if sampling.get('preset',result['sampling']['preset'])=='quick' and 'quick_'+old in sampling else old
        if old_key in sampling and new not in sampling:sampling[new]=sampling[old_key]
    if 'training_domain' in sampling:sampling.setdefault('common_symmetry',sampling['training_domain']=='common_symmetry')
    legacy_keys={'fov_diameter_mm','fov_z_mm','dsv_diameter_mm','source_overrides','requirements',
        'optimization_options','max_evaluations','scan','results','quick','popsize','maxiter','parameters_file','_use_workbench_workflow'}
    canonical={k:v for k,v in raw.items() if k not in legacy_keys}
    result=merge_config(result,canonical);result['schema_version']=SCHEMA_VERSION
    if validate:
        errors=validate_spec(result)
        if errors:raise ValueError('；'.join(errors))
    return result


def legacy_view(canonical, legacy_options=None):
    """Compatibility view for historical disk-only functions, never another input authority.

    Normal CLI commands carry ``canonical_config`` and execute the same workbench
    as the UI. Explicit old flat files retain their old disk workflow and gates.
    """
    cfg=canonicalize_config(canonical)
    result=deepcopy(legacy_options or {})
    for key in ('source_geometry','comsol_root','envelope','manufacturing','search','deliverables',
                'seed','density_kg_m3','chunk_points','polarization_model','evolution'):
        value=deepcopy(cfg[key])
        if isinstance(value,dict) and isinstance(result.get(key),dict):result[key]=merge_config(result[key],value)
        else:result[key]=value
    result['canonical_config']=deepcopy(cfg)
    result['envelope']['half_length_mm']=cfg['envelope']['max_length_mm']/2
    result['envelope']['max_half_height_mm']=cfg['envelope']['max_length_mm']/2
    result['fov_diameter_mm']=cfg['target']['diameter_mm'];result['fov_z_mm']=cfg['target'].get('z_mm',0.)
    result['source_overrides']=dict(cfg['magnet'],enabled=True)
    result['max_evaluations']=cfg['solver']['max_evaluations'];result['maxiter']=cfg['solver']['maxiter']
    result['optimization_options']={k:v for k,v in cfg['solver'].items() if k in ('population_size','initial_local_fraction','initial_sigma_mm','local_refine_evaluations','pair_refine_fraction','pair_max_trials')}
    result['max_validation_candidates']=cfg['max_validation_candidates']
    sample=result.setdefault('sampling',{})
    for prefix in ('','quick_'):
        for old,new in (('training_area','training_volume'),('training_boundary','training_surface'),('validation_area','validation_volume'),('validation_boundary','validation_surface')):
            sample[prefix+old]=cfg['sampling'][new]
    sample.update(refine_starts=cfg['sampling']['refine_starts'],training_domain='common_symmetry' if cfg['sampling']['common_symmetry'] else 'global')
    result.setdefault('popsize',3)
    return result
