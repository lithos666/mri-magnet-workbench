"""Bound-aware perturbations and measured two-coordinate field compensation."""
import math
import numpy as np


def reflect_interval(values, lower, upper):
    """Reflect continuously at hard bounds; never clip probability onto a face.

    A feasible hard boundary necessarily gives an inward mean displacement.
    Reflection removes atoms at the boundary, not this geometric asymmetry.
    """
    lo,hi=float(lower),float(upper)
    values=np.asarray(values,dtype=float)
    if not np.all(np.isfinite(values)) or not all(math.isfinite(x) for x in (lo,hi)) or hi<lo:
        raise ValueError('Invalid reflection values or bounds')
    if hi==lo:return np.full_like(values,lo)
    width=hi-lo
    return lo+width-np.abs(np.remainder(values-lo,2*width)-width)


def compensated_steps(x, probes, low, high, integer, max_trials):
    """Propose local paired directions using measured signed field changes.

    Only probes measured at the supplied x are used. Single-coordinate probes
    provide local secants, never a presumed sign of
    dB/dR. A ppm-improving, field-losing move is paired with a measured field
    gain on a different coordinate. The actual combined design must still be
    solved and pass every original requirement before it can be accepted.
    """
    probes=[p for p in probes if 'origin' in p and np.array_equal(np.asarray(p['origin'],float),x)]
    targets=[p for p in probes if p['delta_field_T']<0 and p['delta_ppm']<0 and not integer[p['index']]]
    gains=[p for p in probes if p['delta_field_T']>0 and not integer[p['index']]]
    targets.sort(key=lambda p:p['delta_ppm']/max(-p['delta_field_T'],1e-30))
    gains.sort(key=lambda p:p['delta_ppm']/max(p['delta_field_T'],1e-30))
    seen=set();made=0
    for target in targets:
        for gain in gains:
            i,j=target['index'],gain['index']
            if i==j:continue
            step=np.zeros_like(x)
            step[i]=target['delta']
            # Small positive field headroom is a proposed move, not a relaxed gate.
            loss=-target['delta_field_T']
            ratio=(loss+max(loss*1e-6,1e-15))/gain['delta_field_T']
            step[j]=gain['delta']*ratio
            factor=1.
            for k in (i,j):
                if step[k]>0:factor=min(factor,(high[k]-x[k])/step[k])
                elif step[k]<0:factor=min(factor,(low[k]-x[k])/step[k])
            if factor<=1e-10:continue
            trial=x+step*min(1.,factor)
            if np.any(trial<low-1e-12) or np.any(trial>high+1e-12):continue
            key=tuple(np.round(trial,14))
            if key in seen:continue
            seen.add(key);made+=1
            yield trial,{'target_index':i,'compensator_index':j,
                'measured_compensator_slope_T':gain['delta_field_T']/gain['delta'],
                'target_delta_field_T':target['delta_field_T'],'step_fraction':min(1.,factor)}
            if made>=max_trials:return
