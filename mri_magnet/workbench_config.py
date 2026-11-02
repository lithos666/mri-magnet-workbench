"""Central, JSON-compatible workbench defaults; legacy CONFIG remains unchanged."""
from copy import deepcopy
import math

DEFAULTS = {
    "mode": "de_novo", "source_geometry": "",
    "target": {"geometry": "sphere", "diameter_mm": 20., "z_mm": 0.,
               "min_field_mT": 30., "max_ppm": 1000.},
    "auxiliary_cylinder": {"enabled": False, "diameter_mm": 20., "length_mm": 20.},
    "envelope": {"protected_bore_diameter_mm": 240., "max_outer_diameter_mm": 384.187454246,
                 "max_length_mm": 500.},
    "magnet": {"cube_mm": 12., "remanence_T": 1.43, "reference_mur": 1.05},
    "manufacturing": {"enabled": True, "layout_mode": "radial_channels",
        "allow_split_pocket_seams": True,
        "pocket_clearance_per_side_mm": .2, "pocket_web_mm": 1.2,
        "inner_wall_mm": 1.2, "outer_wall_mm": 1.2, "base_thickness_mm": 1.6,
        "printer_bed_mm": [480., 480., 255.], "machine_bed_mm": [255., 255., 255.], "sectors_per_layer": 4,
        "bed_margin_per_side_mm": 5., "max_magnet_count": 1736,
        "radial_inner_clearance_mm": 0., "min_central_magnet_gap_mm": 0.,
        "min_axial_module_gap_mm": 0., "bed_fit_mode": "fixed_envelope"},
    "de_novo": {"pair_counts": [5, 6, 7, 8], "column_counts": [24, 26, 28, 30, 32, 34, 36],
                "seed_count": 4, "milp_time_limit_seconds": 3., "transverse_weight": .05,
                "field_margin_ratio": .005, "radial_candidate_count": None, "candidate_pool_size": None, "sparsity_weight": 1e-9,
                "harmonic_order_weights": {"2": 1., "4": 1., "6": .25, "8": .1}},
    "search": {"optimize_radius": True, "optimize_axial_positions": True,
               "axial_shift_limit_mm": 25., "optimize_azimuth_count": False,
               "optimize_radial_ring_count": False, "min_radial_rings_per_layer": 1,
               "max_radial_rings_per_layer": None, "azimuth_count_candidates": [24,26,28,30,32,34,36],
               "topology_limit":4, "azimuth_count_mode":"shared", "stock_aware_column_neighbors":True,
               "seed_subset_max_candidates": 36},
    "scoring": {"mode": "balanced", "field_weight": .5, "field_gain_scale": .10, "ppm_scale": 1000.},
    "sampling": {"preset": "quick", "common_symmetry": True,
        "training_volume": 64, "training_surface": 128,
        "validation_volume": 1024, "validation_surface": 768,
        "refine_starts": 3},
    "solver": {"max_evaluations": 2400, "population_size": 32, "maxiter": 100,
        "max_workers": 1, "local_refine_evaluations": 240, "budget_policy": "strict",
        "pair_refine_fraction": .4, "pair_max_trials": 12,
        "initial_local_fraction": .7, "initial_sigma_mm": .8,
        "halving_factor": 2, "initial_budget_fraction": .3,
        "active_set_evaluations":240,"active_set_trust_radius":.03,"active_set_fd_step":.002},
    "evolution": {"enabled": True, "grid_points": 21, "max_frames": 24,
        "min_refresh_seconds": 5., "generation_interval": 1, "gif_duration_ms": 650},
    "deliverables": {"compile": True, "runtime_validate": False, "air_radius_mm": 650.,
        "air_mu_r": 1., "mesh_size": 5, "timeout_seconds": 300, "class_prefix": "MRI_Design"},
    "comsol_root": "D:/comsol/COMSOL62/Multiphysics",
    "chunk_points": 96, "polarization_model": "remanence", "density_kg_m3": 7500.,
    "seed": 20261101, "max_validation_candidates": 5,
    "analysis": {"slice_grid_size": 21, "integration_volume": 128, "integration_surface": 128,
        "slice_color_limit_ppm": None, "harmonic_max_degree": 8, "max_source_point_pairs": 75000000,
        "mutual_demag": {"enabled": False, "quadrature_order": 1, "max_unknowns": 1200,
            "max_magnets": 10000, "max_matrix_mb": 128., "chunk_points": 48},
        "tolerance": {"enabled": False, "trials": 200, "remanence_half_range_fraction": .015,
            "sector_translation_half_range_mm": .15, "translation_fd_step_mm": .05,
            "max_memory_mb": 192.}},
}


def merge_config(base, changes):
    result = deepcopy(base)
    for key, value in changes.items():
        result[key] = merge_config(result[key], value) if isinstance(value, dict) and isinstance(result.get(key), dict) else deepcopy(value)
    return result


def default_config():
    return deepcopy(DEFAULTS)


def validate_spec(config):
    errors = []
    def positive(value, name, integer=False, zero=False):
        try:
            number = float(value)
            if isinstance(value, bool) or not math.isfinite(number) or (number < 0 if zero else number <= 0) or (integer and number != int(number)):
                raise ValueError()
        except (ValueError, TypeError, OverflowError):
            errors.append(name + ("必须为正整数" if integer else "必须为有限正数"))
    if config.get('mode') not in ('de_novo', 'seeded'): errors.append('设计模式必须为 de_novo 或 seeded')
    target = config.get('target', {})
    if target.get('geometry') not in ('sphere', 'disk'): errors.append('主评价域必须为 sphere 或 disk')
    for key in ('diameter_mm', 'min_field_mT'): positive(target.get(key), 'target.'+key)
    if target.get('max_ppm') is not None: positive(target['max_ppm'], 'target.max_ppm')
    try:
        if not math.isfinite(float(target.get('z_mm', 0.))): raise ValueError()
        if target.get('geometry') == 'sphere' and float(target.get('z_mm', 0.)) != 0.: errors.append('镜像球形 DSV 必须以原点为中心，z_mm=0')
    except (ValueError, TypeError): errors.append('target.z_mm 必须为有限数')
    for key in ('cube_mm', 'remanence_T', 'reference_mur'): positive(config.get('magnet', {}).get(key), 'magnet.'+key)
    for key in ('protected_bore_diameter_mm', 'max_outer_diameter_mm', 'max_length_mm'): positive(config.get('envelope', {}).get(key), 'envelope.'+key)
    for key in ('max_evaluations', 'population_size', 'max_workers', 'maxiter'): positive(config.get('solver', {}).get(key), 'solver.'+key, integer=True)
    if config.get('sampling', {}).get('preset') not in ('quick', 'standard'): errors.append('采样预设必须为 quick 或 standard')
    for key in ('training_volume', 'training_surface', 'validation_volume', 'validation_surface', 'refine_starts'):
        positive(config.get('sampling', {}).get(key), 'sampling.'+key, integer=True)
    for key in ('chunk_points', 'max_validation_candidates'): positive(config.get(key), key, integer=True)
    positive(config.get('density_kg_m3'), 'density_kg_m3')
    if config.get('polarization_model') not in ('remanence', 'isolated_cube_self_demag'): errors.append('未知磁化模型')
    scoring=config.get('scoring',{})
    if scoring.get('mode','ppm') not in ('ppm','balanced'): errors.append('scoring.mode 必须为 ppm 或 balanced')
    for key,default in (('field_gain_scale',.1),('ppm_scale',1000.)):
        positive(scoring.get(key,default),'scoring.'+key)
    try:
        weight=float(scoring.get('field_weight',.5))
        if not math.isfinite(weight) or not 0<=weight<=1: raise ValueError()
    except (ValueError,TypeError): errors.append('scoring.field_weight 必须在0到1之间')
    search=config.get('search',{})
    for key in ('min_radial_rings_per_layer','max_radial_rings_per_layer'):
        if search.get(key) is not None: positive(search[key],'search.'+key,integer=True)
    aux = config.get('auxiliary_cylinder', {})
    if aux.get('enabled'):
        for key in ('diameter_mm', 'length_mm'): positive(aux.get(key), 'auxiliary_cylinder.'+key)
    solver=config.get('solver',{})
    if solver.get('budget_policy','strict') not in ('strict','allow_partial'):
        errors.append('solver.budget_policy 必须为 strict 或 allow_partial')
    seams=config.get('manufacturing',{}).get('allow_split_pocket_seams',False)
    if not isinstance(seams,bool): errors.append('manufacturing.allow_split_pocket_seams 必须为布尔值')
    for key in ('local_refine_evaluations','active_set_evaluations'):
        positive(solver.get(key,0),'solver.'+key,integer=True,zero=True)
    for key in ('active_set_trust_radius','active_set_fd_step'):
        positive(solver.get(key,.03),'solver.'+key)
        try:
            if float(solver.get(key,.03))>1:errors.append('solver.'+key+'必须不大于1')
        except (TypeError,ValueError):pass
    try:
        if float(solver.get('population_size',0))<5:errors.append('solver.population_size至少为5')
    except (TypeError,ValueError):pass
    if config.get('mode')=='de_novo':
        synthesis=config.get('de_novo',{})
        positive(synthesis.get('transverse_weight',.05),'de_novo.transverse_weight',zero=True)
        positive(synthesis.get('field_margin_ratio',.005),'de_novo.field_margin_ratio',zero=True)
        positive(synthesis.get('sparsity_weight',1e-9),'de_novo.sparsity_weight',zero=True)
        positive(synthesis.get('milp_time_limit_seconds',3.),'de_novo.milp_time_limit_seconds')
        if synthesis.get('radial_candidate_count') is not None:
            positive(synthesis['radial_candidate_count'],'de_novo.radial_candidate_count',integer=True)
        if synthesis.get('candidate_pool_size') is not None:
            positive(synthesis['candidate_pool_size'],'de_novo.candidate_pool_size',integer=True)
        weights=synthesis.get('harmonic_order_weights',{})
        if not isinstance(weights,dict):errors.append('harmonic_order_weights 必须为阶次到权重的对象');weights={}
        for order,value in weights.items():
            if str(order) not in ('2','4','6','8'):errors.append('轴向特征阶次只能为2/4/6/8')
            positive(value,'de_novo.harmonic_order_weights.'+str(order),zero=True)
        if synthesis.get('harmonic_max_order') is not None:
            if synthesis['harmonic_max_order'] not in (0,2,4,6,8):errors.append('harmonic_max_order 必须为0/2/4/6/8')
    analysis=config.get('analysis',{})
    if not isinstance(analysis,dict):return errors+['analysis必须为对象']
    for key in ('slice_grid_size','integration_volume','integration_surface','harmonic_max_degree'):
        if key in analysis:positive(analysis[key],'analysis.'+key,integer=True)
    bounds={'slice_grid_size':(5,65),'integration_volume':(1,1024),'integration_surface':(1,1024),'harmonic_max_degree':(2,12)}
    for key,(low,high) in bounds.items():
        try:
            if key in analysis and not low<=float(analysis[key])<=high:errors.append('analysis.'+key+'超出允许范围')
        except (ValueError,TypeError):pass
    if analysis.get('slice_color_limit_ppm') is not None:positive(analysis['slice_color_limit_ppm'],'analysis.slice_color_limit_ppm')
    if 'max_source_point_pairs' in analysis:positive(analysis['max_source_point_pairs'],'analysis.max_source_point_pairs',integer=True)
    for section in ('mutual_demag','tolerance'):
        if not isinstance(analysis.get(section,{}),dict):errors.append('analysis.'+section+'必须为对象');continue
        if not isinstance(analysis.get(section,{}).get('enabled',False),bool):errors.append('analysis.'+section+'.enabled必须为布尔值')
    tolerance=analysis.get('tolerance',{}) if isinstance(analysis.get('tolerance',{}),dict) else {}
    for key in ('trials','translation_fd_step_mm','max_memory_mb'):
        if key in tolerance:positive(tolerance[key],'analysis.tolerance.'+key,integer=key=='trials')
    for key in ('remanence_half_range_fraction','sector_translation_half_range_mm'):
        if key in tolerance:positive(tolerance[key],'analysis.tolerance.'+key,zero=True)
    try:
        if float(tolerance.get('remanence_half_range_fraction',.015))>=1:errors.append('剩磁公差半幅必须小于1')
    except (ValueError,TypeError):pass
    return errors
