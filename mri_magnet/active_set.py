"""Budget-external, finite-difference trust-region Chebyshev field polishing.

All physics is supplied by ``query``. The caller owns the one shared field
budget and manufacturing audit. The LP predicts a step; only an independently
solved nonlinear trial can be accepted.
"""
import math
import numpy as np
from scipy.optimize import linprog
from .objectives import scoring_options


def active_set_polish(x, low, high, integer, query, remaining, weights=None,
                      requirements=None, trust_radius=.03, fd_step=.002, max_steps=3):
    x=np.asarray(x,float).copy();low=np.asarray(low,float);high=np.asarray(high,float)
    integer=np.asarray(integer,bool);req=requirements or {}
    if not (math.isfinite(trust_radius) and 0<trust_radius<=1 and math.isfinite(fd_step) and 0<fd_step<=1):
        raise ValueError('LP trust radius/fd_step must be finite in (0,1]')
    free=np.flatnonzero((high>low)&~integer)
    info={'enabled':True,'state':'not_started','dimensions':len(free),'jacobian_columns':0,
          'trial_steps':0,'accepted_steps':0,'nonlinear_trials':[],
          'method':'finite differences of |B| / weighted mean; trust-region Chebyshev LP',
          'all_physics_queries_charged_by_caller':True,'global_optimum_proven':False}
    if not len(free):info['state']='no_continuous_variables';return x,info
    # One anchor refresh, one directional solve per free coordinate, one trial.
    minimum=len(free)+2;info['required_minimum_evaluations']=minimum
    if remaining()<minimum:info['state']='insufficient_budget_for_full_jacobian';return x,info
    anchor=query(x)
    if anchor is None:info['state']='invalid_anchor';return x,info
    fields=np.asarray(anchor['fields'],float)
    w=np.ones(len(fields)) if weights is None else np.asarray(weights,float)
    if w.shape!=(len(fields),) or not np.all(np.isfinite(w)) or np.any(w<0) or w.sum()<=0:
        raise ValueError('LP quadrature weights must be finite nonnegative and match observer count')
    w=w/w.sum();norm=np.linalg.norm(fields,axis=1);mean=float(w@norm)
    if not math.isfinite(mean) or mean<=1e-12:info['state']='invalid_field_scale';return x,info
    columns=[]
    # The complete Jacobian remains anchored at x until a nonlinear step is
    # accepted. Failed geometry probes cannot be silently treated as zero.
    for index in free:
        width=high[index]-low[index];step=fd_step*width
        sign=1. if high[index]-x[index]>=x[index]-low[index] else -1.
        delta=sign*min(step,(high[index]-x[index]) if sign>0 else (x[index]-low[index]))
        solved=None
        for _ in range(6):
            if abs(delta)<1e-14:break
            probe=x.copy();probe[index]+=delta;solved=query(probe)
            if solved is not None:break
            delta*=.5
        if solved is None:info['state']='incomplete_jacobian_geometry_rejected';return x,info
        columns.append((np.linalg.norm(np.asarray(solved['fields']),axis=1)-norm)/(delta*mean))
        info['jacobian_columns']+=1
    jac=np.column_stack(columns);base=norm/mean;mean_gradient=w@jac
    info['jacobian_coverage']='all active continuous coordinates'
    lo=np.maximum(low[free]-x[free],-trust_radius*(high[free]-low[free]))
    hi=np.minimum(high[free]-x[free],trust_radius*(high[free]-low[free]))
    for attempt in range(int(max_steps)):
        if remaining()<1:info['state']='budget_exhausted_before_trial';break
        # Variables are delta_x, lower_field, upper_field (normalized by the
        # actual anchor mean). Mean and range caps use their linearized changes.
        n=len(free);c=np.r_[np.zeros(n),-1.,1.]
        score=scoring_options(req.get('scoring'))
        if score['mode']=='balanced':
            # First-order version of the SAME engineering loss used by DE.
            # Both relative range and field gain depend on changing the mean.
            fraction=float(np.ptp(norm)/mean);weight=score['field_weight']
            range_weight=(1-weight)/(score['ppm_scale']*1e-6+fraction)
            bmin=max(float(req.get('min_mean_norm_T',mean)),1e-12)
            gain=max(0.,mean/bmin-1.)
            field_weight=weight*(mean/bmin)/(score['field_gain_scale']+gain)
            c=np.r_[-(range_weight*fraction+field_weight)*mean_gradient,-range_weight,range_weight]
            info['linearized_scoring']='balanced log range/mean and fractional field gain'
        matrix=[];rhs=[]
        matrix.extend(np.c_[jac,np.zeros(len(base)),-np.ones(len(base))]);rhs.extend(-base)
        matrix.extend(np.c_[-jac,np.ones(len(base)),np.zeros(len(base))]);rhs.extend(base)
        minimum_mean=max(0.,float(req.get('min_mean_norm_T',0.))-float(req.get('field_tolerance_T',0.)))
        matrix.append(np.r_[-mean_gradient,0.,0.]);rhs.append(1.-minimum_mean/mean)
        if req.get('max_pp_norm_ppm') is not None:
            epsilon=float(req['max_pp_norm_ppm'])*1e-6
            matrix.append(np.r_[-epsilon*mean_gradient,-1.,1.]);rhs.append(epsilon)
        lp=linprog(c,A_ub=np.asarray(matrix),b_ub=np.asarray(rhs),bounds=list(zip(lo,hi))+[(None,None),(None,None)],method='highs',
            options={'primal_feasibility_tolerance':1e-9,'dual_feasibility_tolerance':1e-9})
        if not lp.success:info['state']='linearized_subproblem_infeasible';info['lp_message']=str(lp.message);break
        delta=np.asarray(lp.x[:n]);info['last_predicted_range_fraction']=float(lp.x[-1]-lp.x[-2])
        info['last_linear_objective']=float(lp.fun)
        if np.max(np.abs(delta))<1e-12:info['state']='stationary_linearized_model';break
        trial=x.copy();trial[free]+=delta
        candidate=query(trial);info['trial_steps']+=1
        accepted=candidate is not None and candidate['score']<anchor['score']
        info['nonlinear_trials'].append({'accepted':bool(accepted),'step_norm':float(np.linalg.norm(delta)),
            'metrics':None if candidate is None else candidate['metrics']})
        if accepted:
            x=trial;info['accepted_steps']+=1;info['state']='accepted_independently_verified_step';break
        # Backtrack on the same local linearization; this cannot accept a bad
        # nonlinear step just because its LP prediction looked promising.
        lo*=.5;hi*=.5;info['state']='nonlinear_trial_rejected_trust_region_reduced'
    return x,info
