"""Constructive continuous geometry for one fixed mirror-symmetric topology.

Topology is selected outside this module. Radii have independent ordered slack
coordinates; axial coordinates include actual neighbour gaps and displacement
windows. No initial separating midplanes are imposed.
"""
from __future__ import annotations

from dataclasses import replace
import math
import numpy as np
from .manufacturing import radial_packing_bounds, axial_packing_bounds
from .search_moves import reflect_interval


def packing_for_design(seed, process, envelope):
    if process.get("enabled", True):
        return radial_packing_bounds(seed.cube_mm, process, envelope)
    half=seed.cube_mm/2
    lower=float(envelope["bore_radius_mm"])+half
    upper=math.sqrt(float(envelope["outer_radius_mm"])**2-half**2)-half
    pitch=seed.cube_mm+float(envelope.get("min_gap_mm",0.0))
    return {"minimum_magnet_center_radius_mm":lower,"maximum_magnet_center_radius_mm":upper,
            "required_radial_center_pitch_mm":pitch,
            "maximum_radial_rings":max(0,int(math.floor((upper-lower+1e-9)/pitch))+1),
            "pocket_edge_mm":seed.cube_mm}


def topology_identity(design):
    """Discrete physical architecture; provenance labels do not split runs."""
    groups={}
    for ring in design.rings:groups.setdefault(ring.layer_index,[]).append(ring)
    return tuple((layer,tuple((r.count,r.polarity) for r in sorted(rings,key=lambda r:r.radius_mm)))
                 for layer,rings in sorted(groups.items()))


def encode_slacks(slacks):
    """Deterministic inverse of stick breaking, including zero-slack faces."""
    gaps=np.asarray(slacks,dtype=float)
    if gaps.ndim!=1 or len(gaps)<2 or not np.all(np.isfinite(gaps)):
        raise ValueError("Expected a finite slack vector of length >=2")
    if np.min(gaps)<-1e-8:raise ValueError("Slacks must be non-negative")
    gaps=np.maximum(gaps,0)
    result=np.zeros(len(gaps)-1)
    for index in range(len(result)):
        remaining=float(np.sum(gaps[index:]))
        result[index]=gaps[index]/remaining if remaining>0 else 0.0
    return np.clip(result,0,1)


def decode_slacks(vector,total):
    vector=np.asarray(vector,dtype=float)
    if (vector.ndim!=1 or not np.all(np.isfinite(vector)) or np.any(vector<-1e-12)
            or np.any(vector>1+1e-12) or not math.isfinite(total) or total<-1e-8):
        raise ValueError("Invalid slack coordinates or total")
    gaps=np.zeros(len(vector)+1);remaining=max(0.0,float(total))
    for index,value in enumerate(np.clip(vector,0,1)):
        gaps[index]=value*remaining;remaining-=gaps[index]
    gaps[-1]=max(0.0,remaining)
    return gaps


def radius_slacks(radii,lower,upper,pitch,tolerance=1e-7):
    radii=np.asarray(radii,dtype=float)
    gaps=np.r_[radii[0]-lower,np.diff(radii)-pitch,upper-radii[-1]]
    if np.any(gaps<-tolerance):raise ValueError("Radii violate ordered radial packing bounds")
    return np.maximum(gaps,0)


def _ordered_bounds(lower,upper,separation,tolerance):
    lower,upper=np.asarray(lower,float),np.asarray(upper,float)
    offsets=np.arange(len(lower))*separation
    ell=np.maximum.accumulate(lower-offsets)
    cap=np.minimum.accumulate((upper-offsets)[::-1])[::-1]
    if np.any(ell>cap+tolerance):raise ValueError("Axial displacement windows have no ordered feasible layout")
    return ell,np.maximum(cap,ell),offsets


def get_search_space(seed,manufacturing_config,envelope,search_config):
    """Return fixed-topology variables and constructive vector -> Design builder.

    ``encode_design`` accepts geometrically equivalent topology even when source
    slot labels differ. ``initial_sampler`` is uniform on each radial slack
    simplex; axial samples use sequential conditional intervals, not a claim of
    uniform measure over the entire constrained polytope. Complete manufacture
    auditing remains required before field evaluation.
    """
    process,search=dict(manufacturing_config),dict(search_config)
    tol=float(envelope.get("tolerance_mm",1e-6))
    packing=packing_for_design(seed,process,envelope)
    lower=float(packing["minimum_magnet_center_radius_mm"])
    upper=float(packing["maximum_magnet_center_radius_mm"])
    pitch=float(packing["required_radial_center_pitch_mm"])
    pocket=float(packing["pocket_edge_mm"])
    printing=process.get("enabled",True)
    base=float(process.get("base_thickness_mm",0)) if printing else 0.0
    gap=float(process.get("min_axial_module_gap_mm",0)) if printing else float(envelope.get("min_gap_mm",0))
    shift=float(search.get("axial_shift_limit_mm",12.0))
    if not math.isfinite(shift) or shift<0:raise ValueError("axial_shift_limit_mm must be non-negative and finite")
    if not all(math.isfinite(v) for v in (lower,upper,pitch,pocket,base,gap)) or upper<lower:
        raise ValueError("Invalid physical packing envelope")
    groups={}
    for ring in seed.rings:groups.setdefault(ring.layer_index,[]).append(ring)
    layer_count=len(groups)
    if layer_count<2 or layer_count%2 or set(groups)!=set(range(layer_count)):
        raise ValueError("The seed needs contiguous, nonempty mirror-paired axial layers")
    pair_count=layer_count//2
    pairs=[]
    for pair in range(pair_count):
        positive=sorted(groups[pair],key=lambda r:(r.radius_mm,r.slot_index))
        negative={r.slot_index:r for r in groups[layer_count-1-pair]}
        if len(negative)!=len(positive) or set(negative)!={r.slot_index for r in positive}:
            raise ValueError("Mirror layers need matching unique source slots")
        z=positive[0].z_mm
        for ring in positive:
            other=negative[ring.slot_index]
            if (z<=0 or abs(ring.z_mm-z)>tol or abs(other.z_mm+z)>tol
                    or abs(ring.radius_mm-other.radius_mm)>tol or ring.count!=other.count):
                raise ValueError("Seed positions and counts must be mirror symmetric")
        radii=np.array([r.radius_mm for r in positive])
        slacks=radius_slacks(radii,lower,upper,pitch,tol)
        total=upper-lower-(len(positive)-1)*pitch
        pairs.append({"pair":pair,"positive":positive,"negative":negative,"radii":radii,
                      "z":z,"slacks":slacks,"total":max(0.0,total)})
    z_initial=np.array([pairs[pair]["z"] for pair in range(pair_count-1,-1,-1)])
    separation=pocket+base+gap
    axial = axial_packing_bounds(seed.cube_mm, process, envelope)
    z_min,z_max=axial['minimum_abs_magnet_center_z_mm'],axial['maximum_abs_magnet_center_z_mm']
    if (z_initial[0]<z_min-tol or z_initial[-1]>z_max+tol or np.any(np.diff(z_initial)<separation-tol)):
        raise ValueError("Seed axial modules violate centre gap or axial packing")
    if search.get("optimize_axial_positions",True):
        axial_lower=np.maximum(z_min,z_initial-shift);axial_upper=np.minimum(z_max,z_initial+shift)
    else:axial_lower=z_initial.copy();axial_upper=z_initial.copy()
    ell,cap,offsets=_ordered_bounds(axial_lower,axial_upper,separation,tol)
    variables=[];radial_indices={}
    for pair in pairs:
        indices=[]
        if search.get("optimize_radius",True) and pair["total"]>tol:
            for index,initial in enumerate(encode_slacks(pair["slacks"])):
                indices.append(len(variables))
                variables.append({"name":"radial_slack_pair_%d_gap_%d"%(pair["pair"]+1,index),
                    "lower":0.0,"upper":1.0,"initial":float(initial),"integer":False,
                    "coordinate_kind":"radial_stick_breaking","pair":pair["pair"]+1})
        radial_indices[pair["pair"]]=indices
    axial_indices={};previous=-math.inf
    for index in range(pair_count):
        y=z_initial[index]-offsets[index];lo=max(ell[index],previous);hi=cap[index]
        if cap[index]-ell[index]>tol:
            axial_indices[index]=len(variables)
            variables.append({"name":"axial_slack_pair_%d"%(pair_count-index),"lower":0.0,"upper":1.0,
                "initial":float(np.clip((y-lo)/(hi-lo),0,1)) if hi-lo>tol else 0.0,"integer":False,
                "coordinate_kind":"ordered_axial_window","pair":pair_count-index})
        previous=y
    initial_vector=np.array([v["initial"] for v in variables])

    def build(vector):
        vector=np.asarray(vector,dtype=float)
        if vector.shape!=(len(variables),) or not np.all(np.isfinite(vector)):
            raise ValueError("Parameter vector has wrong shape or non-finite entries")
        if np.any(vector<-1e-10) or np.any(vector>1+1e-10):raise ValueError("Slack parameter outside [0,1]")
        vector=np.clip(vector,0,1)
        if np.array_equal(vector,initial_vector):return seed
        axial=np.empty(pair_count);previous=-math.inf
        for index in range(pair_count):
            lo=max(ell[index],previous);hi=cap[index]
            value=vector[axial_indices[index]] if index in axial_indices else 0.0
            y=lo+value*max(0.0,hi-lo);axial[index]=y+offsets[index];previous=y
        updates={}
        for pair in pairs:
            indices=radial_indices[pair["pair"]]
            if indices:
                slacks=decode_slacks(vector[indices],pair["total"])
                radii=lower+np.arange(len(pair["positive"]))*pitch+np.cumsum(slacks[:-1])
            else:radii=pair["radii"]
            z=axial[pair_count-1-pair["pair"]]
            for ring,radius in zip(pair["positive"],radii):
                updates[(ring.layer_index,ring.slot_index)]=replace(ring,radius_mm=float(radius),z_mm=float(z))
                other=pair["negative"][ring.slot_index]
                updates[(other.layer_index,other.slot_index)]=replace(other,radius_mm=float(radius),z_mm=-float(z))
        return replace(seed,rings=tuple(updates[(r.layer_index,r.slot_index)] for r in seed.rings),name="continuous_candidate")

    def encode_design(design):
        if (design.cube_mm!=seed.cube_mm or design.remanence_T!=seed.remanence_T
                or design.reference_mur!=seed.reference_mur):raise ValueError("Material or dimensions differ")
        if topology_identity(design)!=topology_identity(seed):raise ValueError("Cannot encode a different physical topology")
        actual_groups={}
        for ring in design.rings:actual_groups.setdefault(ring.layer_index,[]).append(ring)
        actual={}
        for layer in groups:
            originals=sorted(groups[layer],key=lambda r:r.radius_mm)
            targets=sorted(actual_groups[layer],key=lambda r:r.radius_mm)
            for expected,ring in zip(originals,targets):
                period=360/ring.count
                if abs((ring.phase_deg-expected.phase_deg+period/2)%period-period/2)>tol:
                    raise ValueError("Cannot encode different physical phase")
                actual[(expected.layer_index,expected.slot_index)]=ring
        vector=initial_vector.copy()
        for pair in pairs:
            radii=np.array([actual[(r.layer_index,r.slot_index)].radius_mm for r in pair["positive"]])
            encoded=encode_slacks(radius_slacks(radii,lower,upper,pitch,tol))
            if radial_indices[pair["pair"]]:vector[radial_indices[pair["pair"]]]=encoded
            elif not np.allclose(radii,pair["radii"],atol=tol,rtol=0):raise ValueError("Radial positions are fixed")
        previous=-math.inf
        for index in range(pair_count):
            pair=pairs[pair_count-1-index];rings=pair["positive"]
            values=[actual[(r.layer_index,r.slot_index)].z_mm for r in rings];z=values[0]
            if any(abs(value-z)>tol for value in values):raise ValueError("Layer has inconsistent z centres")
            if any(abs(actual[(layer_count-1-r.layer_index,r.slot_index)].z_mm+z)>tol
                   or abs(actual[(layer_count-1-r.layer_index,r.slot_index)].radius_mm-actual[(r.layer_index,r.slot_index)].radius_mm)>tol for r in rings):
                raise ValueError("Encoded geometry is not mirror symmetric")
            y=z-offsets[index];lo=max(ell[index],previous);hi=cap[index]
            if y<lo-tol or y>hi+tol:raise ValueError("Axial position violates displacement or neighbour bounds")
            if index in axial_indices:vector[axial_indices[index]]=np.clip((y-lo)/(hi-lo),0,1) if hi-lo>tol else 0.0
            elif abs(y-lo)>tol:raise ValueError("Axial position is fixed")
            previous=y
        return vector

    def initial_sampler(rng,size,include_initial=False):
        if int(size)!=size or size<0:raise ValueError("Population size must be non-negative integer")
        population=rng.uniform(0,1,size=(int(size),len(variables)))
        for pair in pairs:
            indices=radial_indices[pair["pair"]]
            if indices:
                gaps=rng.dirichlet(np.ones(len(indices)+1),size=int(size))*pair["total"]
                for row,slacks in zip(population,gaps):row[indices]=encode_slacks(slacks)
        if include_initial and size:population[0]=initial_vector
        return population

    def local_sampler(rng,size,sigma_mm,center_design=None):
        """Perturb physical radii/Z in mm, then invert the feasible map.

        Subtracting the required pitch gives ordered free positions. Reflecting
        independent Gaussian positions and ordering them keeps radial gaps
        feasible without clamping samples onto walls. Conditional reflected Z
        intervals preserve neighbour spacing and each displacement window.
        Crowded five-ring layers necessarily have smaller realized motion than
        unconstrained layers; report measured RMS instead of promising equality.
        """
        if isinstance(size,bool) or int(size)!=size or size<0:raise ValueError('Invalid local sample size')
        if not math.isfinite(sigma_mm) or sigma_mm<=0:raise ValueError('initial_sigma_mm must be positive')
        size=int(size)
        center_vector=initial_vector if center_design is None else encode_design(center_design)
        center=seed if center_design is None else build(center_vector)
        center_groups={}
        for ring in center.rings:center_groups.setdefault(ring.layer_index,[]).append(ring)
        center_radii={layer:np.array([r.radius_mm for r in sorted(rings,key=lambda r:r.radius_mm)])
                      for layer,rings in center_groups.items()}
        center_z=np.array([center_groups[layer][0].z_mm for layer in range(pair_count-1,-1,-1)])
        population=np.tile(center_vector,(size,1));radial_stats=[];axial_deltas=[]
        for pair in pairs:
            indices=radial_indices[pair['pair']]
            if not indices:continue
            n=len(indices);offset=np.arange(n)*pitch
            base_radii=center_radii[pair['pair']]
            base_positions=base_radii-offset
            upper_free=upper-(n-1)*pitch
            free=reflect_interval(base_positions+rng.normal(0,sigma_mm,size=(size,n)),lower,upper_free)
            physical=np.sort(free,axis=1)+offset
            for row,radii in zip(population,physical):
                row[indices]=encode_slacks(radius_slacks(radii,lower,upper,pitch,tol))
            delta=physical-base_radii
            radial_stats.append({'pair':pair['pair']+1,'ring_count':n,'free_span_mm':pair['total'],
                'rms_delta_R_mm':float(np.sqrt(np.mean(delta*delta))) if size else 0.,
                'max_abs_delta_R_mm':float(np.max(np.abs(delta))) if size else 0.})
        for row in population:
            previous=-math.inf;current=[]
            for index in range(pair_count):
                lo=max(ell[index],previous);hi=cap[index]
                original=center_z[index]-offsets[index]
                y=float(reflect_interval(original+rng.normal(0,sigma_mm),lo,hi)) if index in axial_indices else lo
                if index in axial_indices:row[axial_indices[index]]=(y-lo)/(hi-lo) if hi>lo else 0.
                current.append(y+offsets[index]-center_z[index]);previous=y
            axial_deltas.append(current)
        axial_array=np.asarray(axial_deltas,float)
        local_sampler.last_statistics={'mode':'physical_mm_reflected_ordered_positions','target_sigma_mm':float(sigma_mm),
            'sample_count':size,'radial_pairs':radial_stats,
            'center_parameters':center_vector.tolist(),'center_kind':'seed' if center_design is None else 'provided_feasible_design',
            'axial_windows_remain_seed_windows':True,
            'rms_delta_Z_mm':float(np.sqrt(np.mean(axial_array**2))) if size else 0.,
            'max_abs_delta_Z_mm':float(np.max(np.abs(axial_array))) if size else 0.,
            'boundary_rule':'continuous reflection; feasible boundary motion is inward, not zero-mean',
            'scale_scope':'input Gaussian in physical mm; crowding/order can reduce realized per-ring displacement'}
        return population

    build.local_sampler=local_sampler
    build.encode_design=encode_design;build.initial_sampler=initial_sampler
    build.metadata={"name":"independent radial slack and ordered axial windows","fixed_topology":True,"pair_count":pair_count,
        "radial_packing":packing,"module_thickness_mm":pocket+base,"minimum_module_center_separation_mm":separation,
        "axial_central_minimum_abs_z_mm":z_min,"axial_displacement_limit_mm":shift,"variables":variables,
        "radial_sampling":"Dirichlet(1,...,1) slack simplex then inverse stick breaking",
        "axial_sampling":"uniform sequential feasible conditional coordinates; not globally uniform",
        "local_sampling":"physical-mm reflected Gaussian on ordered free positions; measured displacement statistics",
        "boundary_aliases":True,"initial_vector_exact":True,
        "pair_slots":[[r.slot_index for r in p["positive"]] for p in pairs],
        "radial_free_spans_mm":[p["total"] for p in pairs],
        "axial_window_widths_mm":(axial_upper-axial_lower).tolist(),
        "topology_signature":topology_identity(seed),
        "feasibility":"constructive radius/axial constraints; full manufacturing audit still required"}
    return variables,build
