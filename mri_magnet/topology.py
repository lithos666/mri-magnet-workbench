"""Outer discrete proposals with explicit, count-balanced sampling priors.

Each proposal is a complete fixed topology for continuous optimisation. Source
slot identities are provenance, not arithmetic DE coordinates. The retained
source pool includes slots omitted by the selected manufacturing warm start.
"""
from __future__ import annotations

from dataclasses import replace
from itertools import combinations
import math
import numpy as np

from .manufacturing import make_manufacturing_seed, audit_manufacturing, column_partition_check
from .parameterization import packing_for_design, topology_identity


class TopologyProposals(list):
    """List[Design] with separate auditable proposal-distribution metadata."""
    def __init__(self, designs, metadata, warm_starts=()):
        super().__init__(designs)
        self.metadata=metadata
        self.warm_starts=list(warm_starts)


def sample_ring_counts(rng,lower,upper,size=1):
    """Inclusive discrete uniform draws; no rounded continuous endpoints."""
    lo,hi=np.asarray(lower),np.asarray(upper)
    if (not np.all(np.isfinite(lo)) or not np.all(np.isfinite(hi))
            or np.any(lo!=np.floor(lo)) or np.any(hi!=np.floor(hi))
            or np.any(lo<1) or np.any(hi<lo) or int(size)!=size or size<0):
        raise ValueError("Invalid inclusive integer ring-count bounds")
    shape=np.broadcast(lo,hi).shape
    return rng.integers(lo.astype(int),hi.astype(int)+1,size=(int(size),)+shape)


def _physical_key(design):
    return tuple(sorted((r.layer_index,round(r.radius_mm,9),round(r.z_mm,9),r.count,r.polarity,
                         round(r.phase_deg%(360/r.count),9)) for r in design.rings))


def _polarity(polarity,mode,positive):
    if mode==1 and polarity in ("in","out"):return "out" if polarity=="in" else "in"
    if mode==2 and polarity in ("down","up"):return "up" if polarity=="down" else "down"
    if mode==3:return "down"
    if mode==4:return "out" if positive else "in"
    return polarity


def propose_topologies(source_pool,seed,process,limits,search,rng,limit):
    """Return seed and distinct random architectures, with separate warm starts.

    ``limit`` counts discrete architectures, including the seed. One same-
    architecture source-slot swap may enter ``warm_starts`` without consuming
    a topology quota. Random
    ring counts are inclusive-uniform within each pair's source/packing range;
    same-count subsets are sampled without replacement. Accepted proposals are
    necessarily conditioned by stock and manufacture feasibility. Metadata
    reports warm proposals, rejected draws and prior count histograms separately.
    Optional stock-aware column neighbours reserve one increased-magnet design
    before random proposals; their deterministic metadata is kept separate.
    """
    if int(limit)!=limit or limit<1:raise ValueError("Topology limit must be a positive integer")
    if (source_pool.cube_mm!=seed.cube_mm or source_pool.remanence_T!=seed.remanence_T
            or source_pool.reference_mur!=seed.reference_mur):
        raise ValueError("Source pool and seed must use the same material and magnet dimensions")
    groups={};seed_groups={}
    for ring in source_pool.rings:groups.setdefault(ring.layer_index,{})[ring.slot_index]=ring
    for ring in seed.rings:seed_groups.setdefault(ring.layer_index,{})[ring.slot_index]=ring
    layers=len(groups)
    if layers<2 or layers%2 or set(groups)!=set(range(layers)) or set(seed_groups)!=set(range(layers)):
        raise ValueError("Topology pool and seed need contiguous complete mirror-paired layers")
    pairs=layers//2
    packing=packing_for_design(seed,process,limits)
    minimum=search.get("min_radial_rings_per_layer",2)
    maximum=search.get("max_radial_rings_per_layer")
    if maximum is None:maximum=packing['maximum_radial_rings']
    if (int(minimum)!=minimum or int(maximum)!=maximum or minimum<1 or maximum<minimum):
        raise ValueError("Invalid radial ring-count search range")
    minimum,maximum=int(minimum),int(maximum)
    default_choices=list(range(24,41,2)) if process.get('allow_split_pocket_seams',False) else [24,28,32,36,40]
    choices=tuple(search.get("azimuth_count_candidates",default_choices))
    if (not choices or len(set(choices))!=len(choices)
            or any(not math.isfinite(c) or int(c)!=c or c<3 for c in choices)):
        raise ValueError("Column choices must be unique integers >=3")
    partitions=[column_partition_check(c,process) for c in choices]
    if any(not row['valid'] for row in partitions):
        raise ValueError('; '.join(note for row in partitions for note in row['violations']))
    choices=tuple(int(c) for c in choices)
    count_mode=search.get("azimuth_count_mode","shared")
    if count_mode not in ("shared","per_pair"):raise ValueError("Invalid azimuth_count_mode")
    max_stock=int(process.get("max_magnet_count",2**31-1))
    pools=[];initial_slots=[];initial_counts=[];reference_z=[];reference_phase=[]
    lower_counts=[];upper_counts=[]
    for pair in range(pairs):
        available=tuple(sorted(groups[pair]))
        if set(available)!=set(groups[layers-1-pair]):raise ValueError("Mirror source pools must share slot indices")
        initial=tuple(sorted(seed_groups[pair]))
        if set(initial)!=set(seed_groups[layers-1-pair]) or not set(initial)<=set(available):
            raise ValueError("Seed must be a matching mirror subset of the source pool")
        upper=min(maximum,packing["maximum_radial_rings"],len(available))
        if upper<minimum:raise ValueError("Requested minimum rings exceeds source/packing capacity")
        current=next(iter(seed_groups[pair].values()))
        if len({r.count for r in seed_groups[pair].values()})!=1:
            raise ValueError("One seed layer has inconsistent column counts")
        pools.append(available);initial_slots.append(initial);initial_counts.append(current.count)
        reference_z.append(current.z_mm);reference_phase.append(current.phase_deg)
        lower_counts.append(minimum);upper_counts.append(upper)
    if count_mode=="shared" and len(set(initial_counts))!=1:
        raise ValueError("Shared count mode requires shared seed column counts")
    baseline_audit=audit_manufacturing(seed,process,limits)
    if not baseline_audit["valid"] or seed.magnet_count>max_stock:
        raise ValueError("Topology warm seed must satisfy geometry and stock budget")
    proposals=[seed];seen={topology_identity(seed)};warm_starts=[]
    seen_physical={_physical_key(seed)}
    metadata={"algorithm":"outer categorical proposals; inner fixed-topology geometry",
        "warm_start_count":1,"warm_swap_count":0,"random_draws":0,"random_accepted":0,
        "rejected_stock":0,"rejected_geometry":0,"duplicate_proposals":0,
        "prior":"uniform ring count per pair, uniform subset conditional on count",
        "random_prior_ring_count_histograms":[{} for _ in range(pairs)],
        "source_pool_slots":[list(pool) for pool in pools],"requested_limit":int(limit),
        "stock_neighbor_enabled":bool(search.get("stock_aware_column_neighbors",False)),
        "stock_neighbor_attempts":[],"stock_neighbor_accepted":0,
        "stock_neighbor_policy":"nearest higher column count, minimum mirrored-ring deletions, deterministic stock-aware coverage; not field-ranked"}

    def construct(selected,counts,modes):
        if 2*sum(len(slots)*count for slots,count in zip(selected,counts))>max_stock:
            metadata["rejected_stock"]+=1;return None
        rings=[]
        for pair,(slots,count,mode) in enumerate(zip(selected,counts,modes)):
            phase=(reference_phase[pair]-180/initial_counts[pair]+180/count)%(360/count)
            for positive,layer in ((True,pair),(False,layers-1-pair)):
                for slot in slots:
                    original=groups[layer][slot]
                    inherited=seed_groups[layer].get(slot,original).polarity
                    rings.append(replace(original,count=count,phase_deg=phase,
                        z_mm=reference_z[pair] if positive else -reference_z[pair],
                        polarity=_polarity(inherited,mode,positive)))
        temporary=replace(source_pool,rings=tuple(sorted(rings,key=lambda r:(r.layer_index,r.slot_index))),
                          name="topology_proposal")
        try:
            design,report=make_manufacturing_seed(temporary,process,limits)
        except ValueError:
            metadata["rejected_geometry"]+=1;return None
        if not report["valid"] or len(design.rings)!=len(rings):
            metadata["rejected_geometry"]+=1;return None
        return replace(design,name="topology_proposal")

    def append_unique(design):
        if design is None:return False
        key=topology_identity(design)
        if key in seen:metadata["duplicate_proposals"]+=1;return False
        seen.add(key);seen_physical.add(_physical_key(design));proposals.append(design);return True

    # Guarantee one nearby increased-magnet architecture when explicitly enabled.
    # This is a deterministic coverage proposal, not a sample from the random
    # prior, and not a claim that using more magnet material improves the field.
    # Leaving the flag false executes the original code and RNG path unchanged.
    if search.get("stock_aware_column_neighbors",False) and len(proposals)<limit:
        if not search.get("optimize_azimuth_count",True):
            metadata["stock_neighbor_disabled_reason"]="azimuth column count is fixed"
        else:
            column_neighbours=[]
            if count_mode=="shared":
                column_neighbours=[[value]*pairs for value in sorted(choices) if value>initial_counts[0]]
            else:
                # Per-pair mode changes just one mirror pair at a time. It does
                # not silently impose a shared count on a mixed-count seed.
                moves=sorted((value-initial_counts[pair],value,pair)
                    for pair in range(pairs) for value in choices if value>initial_counts[pair])
                for _,value,pair in moves:
                    counts=list(initial_counts);counts[pair]=value;column_neighbours.append(counts)
            if not column_neighbours:
                metadata["stock_neighbor_disabled_reason"]="no higher allowed column count"
            for counts in column_neighbours:
                ns=[len(slots) for slots in initial_slots]
                attempt={"azimuth_counts_by_pair":list(counts),"source_ring_counts_by_pair":list(ns),
                    "subset_rule":"least squared distance to equally spaced target radii spanning seed radial extent; not field optimized"}
                metadata["stock_neighbor_attempts"].append(attempt)
                if any(n>upper for n,upper in zip(ns,upper_counts)):
                    attempt["status"]="seed counts exceed configured neighbour search cap";continue
                total=lambda:2*sum(n*count for n,count in zip(ns,counts))
                if total()>max_stock and not search.get("optimize_radial_ring_count",True):
                    attempt["status"]="stock limit requires ring deletion but ring counts are fixed";continue
                while total()>max_stock:
                    eligible=[pair for pair,n in enumerate(ns) if n>lower_counts[pair]]
                    if not eligible:break
                    # Each removal saves 2*count magnets. Taking the largest
                    # saving first minimizes the number of ring-pair removals;
                    # ties reduce the most populated layer, then outer first.
                    pair=min(eligible,key=lambda i:(-counts[i],-ns[i],i))
                    ns[pair]-=1
                attempt.update(ring_counts_by_pair=list(ns),magnet_count=total(),
                    deleted_mirrored_ring_pairs=sum(len(slots)-n for slots,n in zip(initial_slots,ns)))
                if total()>max_stock:
                    attempt["status"]="stock limit cannot be met above minimum ring counts";continue
                if total()<=seed.magnet_count:
                    attempt["status"]="no net increase in magnet count within stock limit";continue
                selected=[]
                for pair,n in enumerate(ns):
                    slots=initial_slots[pair]
                    if n==len(slots):selected.append(slots);continue
                    radial={slot:seed_groups[pair][slot].radius_mm for slot in slots}
                    targets=(np.linspace(min(radial.values()),max(radial.values()),n)
                             if n>1 else np.array([(min(radial.values())+max(radial.values()))/2]))
                    # The source/seed has ordered radii; score retained subsets
                    # in physical millimetres, never by source catalog labels.
                    def subset_score(subset):
                        radii=sorted(radial[slot] for slot in subset)
                        return (float(np.sum((np.asarray(radii)-targets)**2)),tuple(subset))
                    selected.append(tuple(min(combinations(slots,n),key=subset_score)))
                attempt["retained_source_slots_by_pair"]=[list(slots) for slots in selected]
                attempt["removed_source_slots_by_pair"]=[sorted(set(old)-set(new))
                    for old,new in zip(initial_slots,selected)]
                # Keep deterministic audit rejections outside random-prior
                # counters. The usual manufacture, stock and geometry checks
                # still run on the complete 14-layer candidate.
                rejected_before={key:metadata[key] for key in ("rejected_stock","rejected_geometry")}
                candidate=construct(selected,counts,[0]*pairs)
                for key,value in rejected_before.items():metadata[key]=value
                if candidate is None:
                    attempt["status"]="manufacturing or geometry rejected";continue
                if append_unique(candidate):
                    attempt.update(status="accepted",topology_index=len(proposals),
                        actual_ring_count=len(candidate.rings),actual_magnet_count=candidate.magnet_count)
                    metadata["stock_neighbor_accepted"]+=1
                    break
                attempt["status"]="duplicate topology"

    # Retain a deliberately labelled high-ring warm neighbour, without claiming
    # it was drawn from the random prior. This restores a discarded L3 option.
    if search.get("optimize_radial_ring_count",True):
        found=False
        for pair in range(pairs):
            for added in sorted(set(pools[pair])-set(initial_slots[pair])):
                for removed in initial_slots[pair]:
                    selected=list(initial_slots)
                    selected[pair]=tuple(sorted((set(selected[pair])-{removed})|{added}))
                    candidate=construct(selected,initial_counts,[0]*pairs)
                    if candidate is None:continue
                    if topology_identity(candidate)==topology_identity(seed):
                        physical=_physical_key(candidate)
                        if physical in seen_physical:continue
                        warm_starts.append(candidate);seen_physical.add(physical)
                        metadata["warm_swap_count"]+=1;found=True;break
                    if len(proposals)<limit and append_unique(candidate):
                        found=True;break
                if found:break
            if found:break
    all_discrete_fixed=not any(search.get(flag,flag!="optimize_magnetization") for flag in (
        "optimize_radial_ring_count","optimize_azimuth_count","optimize_magnetization"))
    attempts=0;max_attempts=max(100,int(limit)*80)
    while len(proposals)<limit and attempts<max_attempts and not all_discrete_fixed:
        attempts+=1;metadata["random_draws"]+=1
        ns=(sample_ring_counts(rng,lower_counts,upper_counts,1)[0]
            if search.get("optimize_radial_ring_count",True) else np.array([len(s) for s in initial_slots]))
        selected=[]
        for pair,n in enumerate(ns):
            histogram=metadata["random_prior_ring_count_histograms"][pair]
            histogram[str(int(n))]=histogram.get(str(int(n)),0)+1
            selected.append(tuple(sorted(rng.choice(pools[pair],size=int(n),replace=False).tolist()))
                            if search.get("optimize_radial_ring_count",True) else initial_slots[pair])
        if search.get("optimize_azimuth_count",True):
            if count_mode=="shared":counts=[choices[int(rng.integers(0,len(choices)))]]*pairs
            else:counts=[choices[int(rng.integers(0,len(choices))) ] for _ in range(pairs)]
        else:counts=initial_counts
        modes=rng.integers(0,5,size=pairs).tolist() if search.get("optimize_magnetization",False) else [0]*pairs
        if append_unique(construct(selected,counts,modes)):metadata["random_accepted"]+=1
    metadata.update(returned_count=len(proposals),attempt_limit_reached=attempts>=max_attempts,
                    topology_identities=[topology_identity(d) for d in proposals],
                    warm_topology_identities=[topology_identity(d) for d in warm_starts])
    return TopologyProposals(proposals,metadata,warm_starts)
