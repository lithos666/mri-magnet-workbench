"""True-scale R-Z placement drawings made from the exported candidate Design.

The drawing takes one representative azimuthal column from each layer.  It is
not a finite-element field plot and does not change the source geometry.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import math
from typing import Any, Mapping

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib import font_manager
import numpy as np


_STYLES = {
    "out": {"arrow": "↑", "color": "#AB2D35", "label": "N 径向朝外", "north_rz": (1.0, 0.0)},
    "in": {"arrow": "↓", "color": "#1955B0", "label": "N 径向朝内", "north_rz": (-1.0, 0.0)},
    "down": {"arrow": "→", "color": "#A56E27", "label": "N 轴向 −Z", "north_rz": (0.0, -1.0)},
    "up": {"arrow": "←", "color": "#22877A", "label": "N 轴向 +Z", "north_rz": (0.0, 1.0)},
}


def _number(value: float, digits: int = 1) -> str:
    if abs(value - round(value)) < 1e-7:
        return str(int(round(value)))
    return f"{value:.{digits}f}"


def _range_label(values: list[float]) -> str:
    if not values:
        return "—"
    lo, hi = min(values), max(values)
    return _number(lo) if math.isclose(lo, hi, abs_tol=1e-7) else f"{_number(lo)}–{_number(hi)}"


def layout_manifest(design: Any) -> dict[str, Any]:
    """Return the exact placement/arrow/count data used by the renderer.

    Labels show the inner radial face (centre radius minus half the edge).
    Both exact centre coordinates and exact display values are retained; only
    the drawing rounds labels. Java/model coordinates remain centre based.
    """
    cube = float(design.cube_mm)
    if not math.isfinite(cube) or cube <= 0:
        raise ValueError("Magnet edge must be positive and finite")
    if not design.rings:
        raise ValueError("Cannot draw a design without rings")
    groups: dict[int, list[Any]] = defaultdict(list)
    seen = set()
    for ring in design.rings:
        if ring.id in seen:
            raise ValueError("Duplicate source ring ID: " + ring.id)
        seen.add(ring.id)
        if ring.polarity not in _STYLES:
            raise ValueError("Unknown polarity: " + ring.polarity)
        if not all(math.isfinite(float(v)) for v in (ring.radius_mm, ring.z_mm, ring.phase_deg)):
            raise ValueError("Layout coordinates must be finite")
        if ring.radius_mm < 0 or int(ring.count) != ring.count or ring.count < 1:
            raise ValueError("Layout radii/counts are invalid")
        groups[int(ring.layer_index)].append(ring)
    half = cube / 2
    layers, ring_rows, radial_gaps, z_values = [], [], [], []
    phases = []
    for layer_index, group in sorted(groups.items()):
        ordered = sorted(group, key=lambda r: (r.radius_mm, r.slot_index))
        z = float(ordered[0].z_mm)
        if any(not math.isclose(float(r.z_mm), z, abs_tol=1e-6) for r in ordered):
            raise ValueError(f"Layer {layer_index} has inconsistent Z coordinates")
        z_values.append(z)
        counts = sorted({int(r.count) for r in ordered})
        layer = {
            "layer_index": layer_index,
            "layer_label": f"R{layer_index + 1:02d}",
            "center_z_mm": z,
            "radial_ring_count": len(ordered),
            "azimuth_counts": counts,
            "magnet_count": sum(int(r.count) for r in ordered),
            "source_ring_ids": [r.id for r in ordered],
        }
        layers.append(layer)
        radial_gaps.extend(float(b.radius_mm - a.radius_mm - cube) for a, b in zip(ordered, ordered[1:]))
        for ring in ordered:
            style = _STYLES[ring.polarity]
            phases.append(float(ring.phase_deg) % 360)
            ring_rows.append({
                "id": ring.id, "layer_index": layer_index,
                "slot_index": int(ring.slot_index),
                "center_z_mm": float(ring.z_mm), "center_r_mm": float(ring.radius_mm),
                "z_min_mm": float(ring.z_mm - half), "z_max_mm": float(ring.z_mm + half),
                "r_min_mm": float(ring.radius_mm - half), "r_max_mm": float(ring.radius_mm + half),
                "display_radius_mm": float(ring.radius_mm - half),
                "cube_mm": cube, "azimuth_count": int(ring.count),
                "phase_deg": float(ring.phase_deg), "polarity": ring.polarity,
                "display_arrow": style["arrow"], "north_rz": list(style["north_rz"]),
                "color": style["color"],
            })
    unique_z = sorted(set(z_values))
    center_pitches = [b-a for a, b in zip(unique_z, unique_z[1:])]
    axial_gaps = [pitch-cube for pitch in center_pitches]
    rmin = min(row["r_min_mm"] for row in ring_rows)
    rmax = max(row["r_max_mm"] for row in ring_rows)
    outer_corner = max(math.hypot(row["r_max_mm"], half) for row in ring_rows)
    common_columns = sorted({row["azimuth_count"] for row in ring_rows})
    phase = phases[0]
    common_phase = all(abs((v-phase+180) % 360-180) < 1e-6 for v in phases)
    return {
        "design_name": str(design.name), "source_sha256": str(getattr(design, "source_sha256", "")),
        "cube_mm": cube, "layers": layers, "rings": ring_rows,
        "layer_count": len(layers), "radial_ring_count": len(ring_rows),
        "magnet_count": sum(row["azimuth_count"] for row in ring_rows),
        "azimuth_counts": common_columns, "common_column_phase": common_phase,
        "positive_z_display_side": "left", "geometry_scale": "equal_mm_in_R_and_Z",
        "radial_label_reference": "inner_radial_face",
        "radial_label_formula": "center_r_mm - cube_mm / 2",
        "radial_label_units": "mm",
        "axial_min_mm": min(row["z_min_mm"] for row in ring_rows),
        "axial_max_mm": max(row["z_max_mm"] for row in ring_rows),
        "axial_span_mm": max(row["z_max_mm"] for row in ring_rows)-min(row["z_min_mm"] for row in ring_rows),
        "axial_center_pitches_mm": center_pitches, "axial_face_gaps_mm": axial_gaps,
        "same_column_radial_face_gaps_mm": radial_gaps,
        "minimum_same_column_radial_face_gap_mm": min(radial_gaps) if radial_gaps else None,
        "radial_face_min_mm": rmin, "radial_face_max_mm": rmax,
        "naked_magnet_bore_diameter_mm": 2*rmin,
        "outer_corner_radius_mm": outer_corner,
        "outer_corner_diameter_mm": 2*outer_corner,
        "scope": "nominal magnet layout; representative azimuthal columns; no field/structural/CAD qualification",
    }



def layout_comparison_manifest(design: Any, reference_design: Any) -> dict[str, Any]:
    """Compare source slots in physical millimetres, never by sorted rank.

    A slot newly restored from the source pool is an addition relative to the
    printable reference; its radius is not subtracted from a different ring.
    Empty source slots and deleted layers remain explicit in the output.
    """
    current, reference = layout_manifest(design), layout_manifest(reference_design)
    def keyed(manifest):
        result = {}
        for row in manifest['rings']:
            key = (row['layer_index'], row['slot_index'])
            if key in result:raise ValueError('Duplicate source layer/slot identity')
            result[key] = row
        return result
    actual, origin = keyed(current), keyed(reference)
    rows = []
    for key in sorted(set(actual)|set(origin)):
        now, before = actual.get(key), origin.get(key)
        status = 'matched' if now is not None and before is not None else 'added' if now is not None else 'removed'
        dr = now['center_r_mm']-before['center_r_mm'] if status=='matched' else None
        dz = now['center_z_mm']-before['center_z_mm'] if status=='matched' else None
        if dr is not None and abs(dr)<1e-10:dr=0.0
        if dz is not None and abs(dz)<1e-10:dz=0.0
        rows.append({'layer_index':key[0],'layer_label':'R%02d'%(key[0]+1),'slot_index':key[1],
            'status':status,'reference_id':before['id'] if before else None,'current_id':now['id'] if now else None,
            'reference_r_mm':before['center_r_mm'] if before else None,
            'current_r_mm':now['center_r_mm'] if now else None,
            'reference_z_mm':before['center_z_mm'] if before else None,
            'current_z_mm':now['center_z_mm'] if now else None,
            'delta_r_mm':dr,'delta_z_mm':dz,'displacement_mm':math.hypot(dr,dz) if dr is not None else None,
            'reference_azimuth_count':before['azimuth_count'] if before else None,
            'current_azimuth_count':now['azimuth_count'] if now else None,
            'reference_polarity':before['polarity'] if before else None,
            'current_polarity':now['polarity'] if now else None,
            'reference':dict(before) if before else None,'current':dict(now) if now else None})
    layers = []
    for index in sorted({row['layer_index'] for row in rows}):
        group=[row for row in rows if row['layer_index']==index]
        old=[row for row in group if row['reference'] is not None]
        new=[row for row in group if row['current'] is not None]
        oz=old[0]['reference_z_mm'] if old else None;nz=new[0]['current_z_mm'] if new else None
        layers.append({'layer_index':index,'layer_label':'R%02d'%(index+1),
            'reference_z_mm':oz,'current_z_mm':nz,
            'delta_z_mm':nz-oz if oz is not None and nz is not None else None,
            'reference_ring_count':len(old),'current_ring_count':len(new),
            'reference_magnet_count':sum(row['reference_azimuth_count'] for row in old),
            'current_magnet_count':sum(row['current_azimuth_count'] for row in new),
            'added_count':sum(row['status']=='added' for row in group),
            'removed_count':sum(row['status']=='removed' for row in group),'rings':group})
    groups={status:[row for row in rows if row['status']==status] for status in ('matched','added','removed')}
    return {'current_design_name':current['design_name'],'reference_design_name':reference['design_name'],
        'matching_rule':'exact (layer_index, slot_index), independent of radial ordering and display IDs',
        'units':'mm','rings':rows,'layers':layers,**groups,
        'matched_count':len(groups['matched']),'added_count':len(groups['added']),'removed_count':len(groups['removed']),
        'moved_count':sum(row['displacement_mm']>1e-9 for row in groups['matched']),
        'current_layout':current,'reference_layout':reference}


def _signed_mm(value: float) -> str:
    return f'{value:+.1f}' if abs(value)>=.05 else '0.0'


def _configure_font() -> None:
    available = {font.name for font in font_manager.fontManager.ttflist}
    names = [name for name in ("Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Arial Unicode MS", "DejaVu Sans") if name in available]
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": names or ["DejaVu Sans"],
                         "axes.unicode_minus": False, "savefig.facecolor": "white"})


def render_rz_layout(design: Any, output_path: str | Path, title: str | None = None,
                     manufacturing_audit: Mapping[str, Any] | None = None,
                     reference_design: Any | None = None,
                     displacement_threshold_mm: float = .5,
                     reference_label: str = 'PAVA起点') -> Path:
    """Save a true-scale R-Z placement diagram as PNG or PDF.

    Pass a .pdf path for a vector PDF; .png is the default.  The renderer uses
    only the supplied candidate's dimensions, positions, polarities and counts.
    """
    manifest = layout_manifest(design)
    if not math.isfinite(displacement_threshold_mm) or displacement_threshold_mm<0:
        raise ValueError('displacement_threshold_mm must be non-negative and finite')
    comparison = layout_comparison_manifest(design,reference_design) if reference_design is not None else None
    reference = comparison['reference_layout'] if comparison else None
    target = Path(output_path).resolve()
    if target.suffix.lower() not in (".png", ".pdf", ".svg"):
        raise ValueError("Layout output must be PNG, PDF or SVG")
    target.parent.mkdir(parents=True, exist_ok=True)
    _configure_font()
    fig = plt.figure(figsize=(25.6, 10), facecolor="white")
    ax = fig.add_axes([.060, .375, .915, .425])
    table = fig.add_axes([.060, .155, .915, .17], sharex=ax)
    table.set_ylim(0, 1)
    table.axis("off")
    cube = manifest["cube_mm"]
    z_min, z_max = manifest["axial_min_mm"], manifest["axial_max_mm"]
    if reference is not None:
        z_min=min(z_min,reference['axial_min_mm']);z_max=max(z_max,reference['axial_max_mm'])
    z_span = max(z_max-z_min, cube)
    padding = max(25.0,z_span*.045) if comparison else max(10.0,z_span*.025)
    # Reverse the axis so increasing negative Z runs to the right, matching
    # the placement reference and the meaning of the horizontal arrow glyphs.
    ax.set_xlim(z_max+padding, z_min-padding)
    r_min, r_max = manifest["radial_face_min_mm"], manifest["radial_face_max_mm"]
    visible_r_min=min(r_min,reference['radial_face_min_mm']) if reference else r_min
    visible_r_max=max(r_max,reference['radial_face_max_mm']) if reference else r_max
    y_min = min(110.0, visible_r_min-7.0) if visible_r_min >= 100 else max(0.0, visible_r_min-7.0)
    y_max = max(200.0, visible_r_max+7.0) if visible_r_max <= 200 else visible_r_max+7.0
    ax.set_ylim(y_min, y_max)
    ax.set_aspect("equal", adjustable="box")
    tick_start = math.ceil(y_min/10)*10
    ticks = np.arange(tick_start, y_max+1e-7, 20)
    if len(ticks) < 3:
        ticks = np.linspace(y_min, y_max, 4)
    ax.set_yticks(ticks)
    ax.tick_params(axis="y", labelsize=11, colors="#64748B", length=0, pad=10)
    ax.tick_params(axis="x", labelbottom=False, bottom=False)
    ax.set_ylabel("R [mm]", fontsize=11, rotation=0, loc="top", labelpad=18, color="#64748B")
    ax.grid(axis="y", color="#DBE2EB", linewidth=.8)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.axvline(0, color="#A8A8A8", linestyle=(0, (5, 5)), linewidth=1.1)
    ax.text(0, y_min-3.5, "Z=0", ha="center", va="top", fontsize=10, color="#64748B", clip_on=False)
    if comparison:
        for entry in comparison['rings']:
            before=entry['reference']
            if before is None:continue
            deleted=entry['status']=='removed'
            ax.add_patch(Rectangle((before['z_min_mm'],before['r_min_mm']),before['cube_mm'],before['cube_mm'],
                facecolor='#B7BEC7' if deleted else 'none',alpha=.16 if deleted else .9,
                edgecolor='#8793A2',linestyle=(0,(3,2)),linewidth=.9,zorder=1))
            if deleted:
                ax.text(before['center_z_mm'],before['center_r_mm'],'删',ha='center',va='center',
                    fontsize=8,color='#8793A2',zorder=2)
    for row in manifest["rings"]:
        ax.add_patch(Rectangle((row["z_min_mm"], row["r_min_mm"]), cube, cube,
                               facecolor=row["color"], edgecolor="white", linewidth=.65))
        ax.text(row["center_z_mm"], row["center_r_mm"]+2.6,
                row["display_arrow"], color="white", ha="center", va="center", fontsize=14)
        ax.text(row["center_z_mm"], row["center_r_mm"]-2.1,
                _number(row["display_radius_mm"]), color="white", ha="center", va="center", fontsize=10.5)
    nearest_pitch = min(manifest["axial_center_pitches_mm"], default=40.0)
    compact = nearest_pitch < 23
    # A close central pair must not shrink every outer layer's dimensions.
    centers=sorted({row['center_z_mm'] for row in manifest['layers']})
    local_compact={z:min((abs(z-other) for other in centers if other!=z),default=40.)<23 for z in centers}
    if comparison:
        # The thin arrow joins actual source-slot centres. Per-ring radial
        # labels live beside the cubes; shared axial movement is written once
        # per layer below, keeping magnetic arrow/centre-radius labels readable.
        for entry in comparison['matched']:
            if entry['displacement_mm']<=displacement_threshold_mm:continue
            before,now=entry['reference'],entry['current']
            ax.annotate('',xy=(now['center_z_mm'],now['center_r_mm']),
                xytext=(before['center_z_mm'],before['center_r_mm']),
                arrowprops={'arrowstyle':'-|>','color':'#D79221','lw':.9,'mutation_scale':6,
                            'alpha':.85,'shrinkA':0,'shrinkB':0},zorder=4)
            if abs(entry['delta_r_mm'])>displacement_threshold_mm and not local_compact.get(now['center_z_mm'],False):
                ax.text(now['center_z_mm']-now['cube_mm']/2-1.,now['center_r_mm'],
                    'ΔR'+_signed_mm(entry['delta_r_mm']),ha='left',va='center',fontsize=7,
                    color='#725018',zorder=5,
                    bbox={'facecolor':'white','edgecolor':'none','alpha':.86,'pad':.35})
        for entry in comparison['added']:
            now=entry['current']
            ax.text(now['center_z_mm']-now['cube_mm']/2-1.,now['center_r_mm'],'新增',
                ha='left',va='center',fontsize=7,color='#087F73',zorder=5)
    layer_deltas={row['layer_index']:row for row in comparison['layers']} if comparison else {}
    for layer in manifest["layers"]:
        z = layer["center_z_mm"]
        label_font = 8.0 if local_compact[z] else 12.0
        count_label = "/".join(str(n) for n in layer["azimuth_counts"])
        table.text(z, .94 if comparison else .91, layer["layer_label"], ha="center", va="center", fontsize=label_font+1, weight="bold", color="#172B4D")
        table.text(z, .72 if comparison else .66, "Z="+("+" if z > 0 else "")+_number(z), ha="center", va="center", fontsize=label_font, color="#334155")
        if comparison:
            delta=layer_deltas[layer['layer_index']]['delta_z_mm']
            delta_text='ΔZ '+_signed_mm(delta) if delta is not None else '新增层'
            table.text(z,.50,delta_text,ha='center',va='center',fontsize=label_font-2,color='#8C601E')
        table.text(z, .28 if comparison else .42, f"{count_label}列 × {layer['radial_ring_count']}环", ha="center", va="center", fontsize=label_font-1, color="#64748B")
        table.text(z, .06 if comparison else .18, f"{layer['magnet_count']}块", ha="center", va="center", fontsize=label_font, color="#64748B")
    fig.text(.027, .945, title or f"MRI · {manifest['layer_count']} 层磁铁放置尺寸", fontsize=23, weight="bold", color="#172B4D")
    column_description = (f"每个色块绕圆周重复 {manifest['azimuth_counts'][0]} 次"
                          if len(manifest["azimuth_counts"]) == 1 else "各层周向列数详见底部")
    section_description = "通过共同周向列中心的 R–Z 剖面" if manifest["common_column_phase"] else "每层取代表周向列的 R–Z 展开图"
    fig.text(.027, .897, f"{section_description} · 等比例名义几何 · 单位 mm · {column_description}", fontsize=13, color="#64748B")
    fig.text(.027, .854, f"方块 {_number(cube)}×{_number(cube)}×{_number(cube)} · 数字为底边半径 R底=R中心−边长/2 [mm] · ↑ N朝外　↓ N朝内　→ N朝−Z　← N朝+Z", fontsize=12, color="#334155")
    if comparison:
        detail='紧凑层位仅用箭头表示位移，精确ΔR见比较数据。' if compact else '侧注ΔR、下表ΔZ，单位mm；位移超过'+_number(displacement_threshold_mm)+'才标箭头。'
        fig.text(.027,.815,'灰色虚线：'+reference_label+'   实心色块：当前方案   橙色箭头：中心位移   淡灰色：已删除   '+detail,
                 fontsize=10.5,color='#725018')
    axial_label = (f"磁块轴向总长 {_number(manifest['axial_span_mm'])} · 中心距 {_range_label(manifest['axial_center_pitches_mm'])}"
                   f" · 轴向面隙 {_range_label(manifest['axial_face_gaps_mm'])}")
    radial_gap = manifest["minimum_same_column_radial_face_gap_mm"]
    if radial_gap is not None:
        axial_label += f" · 同列径向最小面隙 {_number(radial_gap)}"
    fig.text(.5, .115, axial_label+" mm", ha="center", fontsize=12, color="#334155")
    fig.text(.027, .067,
        f"裸磁体通孔 Φ{_number(manifest['naked_magnet_bore_diameter_mm'])} · 径向面包络 {_number(r_min)}–{_number(r_max)}"
        f" · 最外角实际半径 {manifest['outer_corner_radius_mm']:.3f} · 合计 {manifest['radial_ring_count']} 环 / {manifest['magnet_count']} 块。",
        fontsize=11, color="#64748B")
    note = "色块数字为径向内侧底边半径，Z为中心坐标，均按可读精度显示；精确中心坐标见同方案 JSON/CSV。图示名义磁块尺寸，侧壁和槽端余量另计。"
    if manufacturing_audit is not None:
        passed = bool(manufacturing_audit.get("valid", False))
        note += " 当前打印几何初筛："+("通过" if passed else "未通过")+"；未包含结构强度验收。"
    fig.text(.027, .026, note, fontsize=10, color="#64748B")
    # Equal-mm aspect can narrow the plotting box for shorter assemblies.
    # Align metadata to that actual box rather than to the nominal full width.
    fig.canvas.draw()
    actual_box = ax.get_position()
    table.set_position([actual_box.x0, .155, actual_box.width, .17])
    fig.savefig(target, dpi=100 if target.suffix.lower() == ".png" else 150, facecolor="white")
    plt.close(fig)
    return target
