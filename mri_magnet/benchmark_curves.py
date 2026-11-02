"""Two axes for topology comparisons, reconstructed without extra field calls."""
from pathlib import Path
from .objectives import feasibility_score, check_performance
from .serialization import save_json


def optimization_curves(result, requirements):
    records=result.get('records',[])
    by_id={r['candidate_id']:r for r in records}
    best=None; curve=[]
    for row in records:
        if best is None or feasibility_score(row['metrics'],requirements)<feasibility_score(best['metrics'],requirements):best=row
        curve.append({'evaluations':row.get('global_evaluations',row.get('field_evaluations')),
            'topology':row.get('topology_index',1),'candidate_id':best['candidate_id'],
            'metrics':best['metrics'],'feasible':check_performance(best['metrics'],requirements)['valid']})
    primary=[]
    runs=result.get('metadata',{}).get('runs',[])
    if runs:
        metadata=runs[0].get('metadata',{})
        initial=next((s for s in metadata.get('stage_summaries',[]) if s.get('stage')=='initialization'),None)
        if initial:primary.append({'generation':0,'evaluations':initial.get('field_evaluations'),
            'candidate_id':initial.get('candidate_id'),'metrics':initial['metrics']})
        for log in metadata.get('iteration_logs',[]):
            if log.get('event')!='generation':continue
            row=by_id.get(log.get('best_candidate_id'))
            metrics=log.get('best_metrics') or (row or {}).get('metrics')
            if metrics:primary.append({'generation':log['generation'],'evaluations':log.get('field_evaluations_total'),
                'candidate_id':log.get('best_candidate_id'),'metrics':metrics})
    return {'total_evaluations':curve,'primary_de_generations':primary,
        'primary_generation_scope':'initialization and completed DE generations only; excludes later coordinate/paired polish',
        'missing_evaluation_counters':any(r['evaluations'] is None for r in curve),
        'extra_field_evaluations':0}


def compare_curves(rows, public_dir):
    comparisons=[]
    for seed in sorted({r['optimizer_seed'] for r in rows}):
        modes={r['mode']:r for r in rows if r['optimizer_seed']==seed}
        if set(modes)!={'fixed','mixed'}:continue
        fixed={x['generation']:x for x in modes['fixed'].get('curves',{}).get('primary_de_generations',[])}
        mixed={x['generation']:x for x in modes['mixed'].get('curves',{}).get('primary_de_generations',[])}
        common=sorted(set(fixed)&set(mixed))
        fixed_total=[p for p in modes['fixed'].get('curves',{}).get('total_evaluations',[]) if p['evaluations'] is not None]
        mixed_total=[p for p in modes['mixed'].get('curves',{}).get('total_evaluations',[]) if p['evaluations'] is not None]
        common_cost=min(max(p['evaluations'] for p in fixed_total),max(p['evaluations'] for p in mixed_total)) if fixed_total and mixed_total else None
        at_cost={mode:next((p for p in reversed(points) if p['evaluations']<=common_cost),None)
            for mode,points in [('fixed',fixed_total),('mixed',mixed_total)]} if common_cost is not None else {}
        population_hashes=[modes[m].get('primary_initial_population_sha256') for m in ('fixed','mixed')]
        comparisons.append({'optimizer_seed':seed,'common_primary_generations':common,
            'common_actual_evaluation_limit':common_cost,'matched_actual_evaluation_rows':at_cost,
            'same_primary_initial_population':population_hashes[0]==population_hashes[1] if all(population_hashes) else None,
            'matched_primary_generation_rows':[{'generation':g,'fixed':fixed[g],'mixed':mixed[g]} for g in common],
            'scope':'training metrics at matched complete primary-topology DE generations; no additional optimization or validation tuning'})
    path=Path(public_dir)/'双口径收敛曲线.json'
    save_json(path,{'rows':[{k:r.get(k) for k in ('mode','optimizer_seed','curves')} for r in rows],
        'comparisons':comparisons,'equal_total_budget_is_separate_from_equal_primary_depth':True})
    if not any(r.get('curves',{}).get('primary_de_generations') for r in rows):return comparisons,None
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from .layout import _configure_font
    _configure_font();fig,axes=plt.subplots(1,2,figsize=(14,5.2))
    for row in rows:
        label='%s / %d'%(row['mode'],row['optimizer_seed'])
        first=[p for p in row.get('curves',{}).get('total_evaluations',[]) if p['evaluations'] is not None and p['feasible']]
        second=row.get('curves',{}).get('primary_de_generations',[])
        if first:axes[0].step([p['evaluations'] for p in first],[p['metrics']['pp_norm_ppm'] for p in first],where='post',label=label)
        if second:axes[1].plot([p['generation'] for p in second],[p['metrics']['pp_norm_ppm'] for p in second],marker='.',label=label)
    axes[0].set(xlabel='累计正式搜索求场次数',ylabel='可行最好值 / ppm',title='相同总预算：所有拓扑、DE及局部搜索')
    axes[1].set(xlabel='主拓扑已完成DE代数',ylabel='主拓扑当时最好值 / ppm',title='相同主拓扑代数：只比较共同代数范围')
    for ax in axes:
        ax.set_yscale('symlog',linthresh=.01);ax.grid(alpha=.2)
        if ax.lines:ax.legend(fontsize=8)
    fig.tight_layout();image=Path(public_dir)/'双口径收敛曲线.png';fig.savefig(image,dpi=140);plt.close(fig)
    return comparisons,image.name
