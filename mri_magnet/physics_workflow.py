"""Shared V6 preparation, candidate ranking and bounded cost preflight."""
from copy import deepcopy
import time
from .geometry import Design
from .metrics import evaluate_design
from .objectives import check_performance,feasibility_score


def evaluate_candidates(rows,config,sample,requirements,engine,limit=None,cancel_check=None):
    """Re-evaluate every proposed basin before official seed selection."""
    from .field_models import model_fingerprint,evaluation_fingerprint
    evaluated=[]
    for original in rows:
        if cancel_check:cancel_check()
        row=deepcopy(original);design=Design.from_dict(row['design'])
        row['proposal_metrics']=deepcopy(row['metrics'])
        started=time.perf_counter();before=dict(engine.statistics)
        try:
            metrics,_=evaluate_design(design,engine,sample,config['density_kg_m3'])
        except Exception as error:
            row.update(physics_status='failed',physics_error=str(error),performance_valid=False)
            evaluated.append(row)
            continue
        metrics.update(field_model=engine.polarization_model,model_fingerprint=model_fingerprint(config),
                       evaluation_fingerprint=evaluation_fingerprint(config,sample))
        row.update(metrics=metrics,performance=check_performance(metrics,requirements),
                   selection_score=feasibility_score(metrics,requirements),physics=deepcopy(config.get('physics',{})))
        row['performance_valid']=row['performance']['valid']
        row['physics_cost']={'elapsed_seconds':time.perf_counter()-started,
            'engine_statistics':{k:v-before.get(k,0) for k,v in engine.statistics.items() if isinstance(v,(int,float))}}
        evaluated.append(row)
    ordered=sorted((r for r in evaluated if r.get('physics_status')!='failed'),key=lambda row:(row['selection_score'],row['id']))
    count=len(ordered) if limit is None else int(limit)
    selected=[];styles=set()
    for row in ordered:
        if row.get('style') not in styles and len(selected)<count:
            selected.append(row);styles.add(row.get('style'))
    selected_ids={row['id'] for row in selected}
    for row in ordered:
        if row['id'] not in selected_ids and len(selected)<count:selected.append(row)
    return selected,evaluated


def physics_preflight(config,seeds,budget_plan):
    from .field_models import model_fingerprint,response_options
    from .material_response import preflight_response
    coupled=config.get('physics',{}).get('field_model')=='coupled'
    workers=int(config['solver']['max_workers']);rows=[];violations=[]
    for index,seed in enumerate(seeds):
        design=Design.from_dict(seed['design'])
        check=preflight_response(design,response_options(config)) if coupled else {'valid':True,'violations':[],
            'vector_unknowns':0,'estimated_working_bytes':0}
        cost=seed.get('physics_cost',{});seconds=cost.get('elapsed_seconds')
        rows.append(dict(check,island=index+1,name=seed.get('style',seed['id']),measured_evaluation_seconds=seconds))
        violations.extend('岛%d: %s'%(index+1,reason) for reason in check['violations'])
    times=[r['measured_evaluation_seconds'] for r in rows if r['measured_evaluation_seconds'] is not None]
    estimate=sum(times)/len(times)*config['solver']['max_evaluations']/min(workers,len(rows)) if times else None
    peak=sorted((r['estimated_working_bytes'] for r in rows),reverse=True)[:workers]
    available=None
    try:
        import os,ctypes
        if os.name=='nt':
            class MemoryStatus(ctypes.Structure):
                _fields_=[('length',ctypes.c_ulong),('load',ctypes.c_ulong),
                    ('total_physical',ctypes.c_ulonglong),('available_physical',ctypes.c_ulonglong),
                    ('total_page',ctypes.c_ulonglong),('available_page',ctypes.c_ulonglong),
                    ('total_virtual',ctypes.c_ulonglong),('available_virtual',ctypes.c_ulonglong),
                    ('available_extended',ctypes.c_ulonglong)]
            status=MemoryStatus();status.length=ctypes.sizeof(status)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):available=int(status.available_physical)
    except (ImportError,AttributeError,OSError):pass
    if available is not None and sum(peak)>.7*available:
        violations.append('预计并行互耦工作内存超过当前可用内存的70%；请减少并行数或关闭占用程序')
    return {'valid':not violations,'violations':violations,'field_model':'coupled' if coupled else 'legacy',
        'model_fingerprint':model_fingerprint(config),'per_island':rows,
        'max_unknowns':max((r['vector_unknowns'] for r in rows),default=0),
        'estimated_parallel_working_bytes':sum(peak),'estimated_seconds':estimate,
        'available_physical_bytes':available,
        'estimate_scope':'Measured seed evaluation extrapolation; topology, cache and system load change runtime; excludes synthesis, plotting, dense and FEM',
        'sampling_and_preparation_budgets_separate':True}
