"""基于已计算数组生成 PNG 和独立中文 HTML；此模块不调用求场器。"""

from __future__ import annotations

import base64
import html
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence
from urllib.parse import quote

from .serialization import jsonable as _json_safe
from .run_overview import best_record, validation_label
from .objectives import pareto_front
from .design_contract import check_performance, record_metrics, eligible_records, valid_record_metrics

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import Normalize
from matplotlib.patches import Patch, Rectangle, Circle
from matplotlib.ticker import ScalarFormatter
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np


METRIC_LABELS = {
    "mean_norm_T": ("平均 |B|", "mT", 1000.0),
    "pp_norm_ppm": ("|B| 峰峰不均匀度", "ppm", 1.0),
    "rms_norm_ppm": ("|B| RMS 不均匀度", "ppm", 1.0),
    "center_Bz_T": ("中心有符号 Bz", "mT", 1000.0),
    "mean_Bz_T": ("平均有符号 Bz", "mT", 1000.0),
    "pp_Bz_ppm": ("Bz 峰峰不均匀度", "ppm", 1.0),
    "max_transverse_T": ("最大横向场", "mT", 1000.0),
    "magnet_count": ("磁块数量", "块", 1.0),
    "mass_magnet_kg": ("永磁材料质量", "kg", 1.0),
}
PARAMETER_LABELS = {
    "radial_step_mm": "径向环中心距 [mm]",
    "radial_pitch_mm": "径向环中心距 [mm]",
    "axial_pitch_mm": "轴向环中心距 [mm]",
    "layer_pitch_mm": "轴向层中心距 [mm]",
    "radius_offset_mm": "磁环半径增量 [mm]",
    "end_radius_offset_mm": "端部磁环半径增量 [mm]",
    "ring_pitch_mm": "轴向环中心距 [mm]",
    "gap_mm": "磁块表面净间隙 [mm]",
    "remanence_T": "剩磁 Br [T]",
    "R05-L1_radius_mm": "R05 / L1 磁环中心半径 [mm]",
    "central_pair_abs_z_mm": "中央镜像磁环 |z| [mm]",
    "magnets_per_ring": "每环磁块数量",
    "z_pair_7_mm": "中央层中心距的一半 [mm]",
}


def _configure_style() -> None:
    installed = {entry.name for entry in font_manager.fontManager.ttflist}
    fonts = [name for name in ("Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Arial Unicode MS") if name in installed]
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": fonts + ["DejaVu Sans"],
        "axes.unicode_minus": False,
        "figure.facecolor": "white",
        "axes.facecolor": "#f8fafc",
        "axes.edgecolor": "#cbd5e1",
        "axes.titleweight": "bold",
        "axes.labelcolor": "#334155",
        "text.color": "#172b4d",
        "xtick.color": "#475569",
        "ytick.color": "#475569",
        "grid.color": "#dbe4ee",
        "grid.alpha": 0.7,
        "savefig.facecolor": "white",
        "font.size": 10,
    })


def _number(value: Any) -> Optional[float]:
    try:
        number = float(value)
        return number if np.isfinite(number) else None
    except (TypeError, ValueError):
        return None


def _fmt(value: Any, digits: int = 4) -> str:
    number = _number(value)
    if number is None:
        return "—"
    if number.is_integer() and abs(number) < 1000000:
        return format(number, ",.0f")
    if number != 0 and (abs(number) < 0.001 or abs(number) >= 1000000):
        return format(number, ".3g")
    return format(number, ",." + str(digits) + "f").rstrip("0").rstrip(".")


def _esc(value: Any) -> str:
    return html.escape(str(value), quote=True)


def _is_planar(payload: Mapping[str, Any]) -> bool:
    return payload.get("model_info", {}).get("fov_geometry") == "disk"


def _metric_label(key: str, payload: Optional[Mapping[str, Any]] = None) -> str:
    label = METRIC_LABELS[key][0]
    if payload and _is_planar(payload):
        if key == "mean_norm_T":
            return "FOV 面积平均 |B|"
        if key == "mean_Bz_T":
            return "FOV 面积平均有符号 Bz"
        if key == "rms_norm_ppm":
            return "FOV 面积 RMS 不均匀度"
    return label


def _comsol_status(model_info: Mapping[str, Any]) -> str:
    state = model_info.get("comsol_status", model_info.get("comsol_validation_status", model_info.get("comsol", "未提供状态")))
    if isinstance(state, Mapping):
        state = state.get("status", "未提供状态")
    labels = {
        "not_run": "尚未执行 FEM 对标", "not_performed": "尚未执行 FEM 对标",
        "not_compared": "尚未完成 FEM 对标", "not_validated": "尚未完成 FEM 验证",
        "pending": "FEM 对标待完成", "unverified": "FEM 一致性尚未验证",
        "exported_only": "仅导出对标输入，尚未求解 FEM",
    }
    return labels.get(str(state), str(state))


def _metric(metrics: Mapping[str, Any], key: str) -> Optional[float]:
    value = _number(metrics.get(key))
    return None if value is None else value * METRIC_LABELS[key][2]


def _parameter_label(name: str) -> str:
    if name.startswith("radial_fill_pair_"):
        return "径向重排系数（-1 靠内，+1 靠外）"
    return PARAMETER_LABELS.get(name, name + (" [mm]" if name.endswith("_mm") else ""))


def _baseline_parameter(baseline: Mapping[str, Any], name: str) -> Optional[float]:
    for mapping in (baseline.get("parameters", {}), baseline.get("design", {}), baseline):
        if isinstance(mapping, Mapping) and name in mapping:
            return _number(mapping[name])
    return None


def _scan_reference(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    return payload.get("scan_reference") or payload.get("baseline", {})


def _manufacturing_audit(row: Mapping[str, Any], payload: Mapping[str, Any]) -> Optional[Mapping[str, Any]]:
    if isinstance(row.get("manufacturing_check"), Mapping):
        return row["manufacturing_check"]
    manufacturing = payload.get("manufacturing", {})
    ident = str(row.get("candidate_id", ""))
    if ident == str(payload.get("baseline", {}).get("candidate_id", "baseline")):
        return manufacturing.get("baseline_audit")
    seed = manufacturing.get("seed_record", {})
    if seed and ident == str(seed.get("candidate_id", "printable_seed")):
        return manufacturing.get("seed_audit")
    return None


def _manufacturing_label(audit: Optional[Mapping[str, Any]]) -> str:
    if not audit or "valid" not in audit:
        return "新增制造约束未审查"
    if audit.get("checks", {}).get("manufacturing_enabled") is False:
        return "制造检查已关闭；裸磁块及数量检查" + ("通过" if audit.get("valid") else "未通过")
    return "几何工艺初筛通过" if bool(audit["valid"]) else "几何工艺初筛未通过"


def _original_label(payload: Mapping[str, Any]) -> str:
    audit = payload.get("manufacturing", {}).get("baseline_audit")
    return "原始物理对照（制造未通过）" if audit and audit.get("valid") is False else "原始物理对照"


def _save(fig: Any, output_dir: Path, filename: str) -> Path:
    path = output_dir / filename
    fig.savefig(path, dpi=155, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_geometry(output_dir: Path, payload: Mapping[str, Any]) -> Optional[Path]:
    arrays = payload.get("magnet_arrays", {})
    positions = np.asarray(arrays.get("positions_mm", []), dtype=float)
    vectors = np.asarray(arrays.get("north_vectors", []), dtype=float)
    if positions.size == 0:
        return None
    positions = positions.reshape(-1, 3)
    if vectors.shape != positions.shape:
        raise ValueError("north_vectors 应与 positions_mm 同为 N×3 数组")
    fig = plt.figure(figsize=(11, 8.5))
    ax = fig.add_subplot(111, projection="3d")
    radial_dot = np.sum(positions[:, :2] * vectors[:, :2], axis=1)
    colors = np.where(radial_dot >= 0, "#dd624e", "#3c82b4")
    cube = _number(arrays.get("cube_mm", payload.get("baseline", {}).get("design", {}).get("cube_mm")))
    angles = np.asarray(arrays.get("radial_angles_deg", []), dtype=float).reshape(-1)
    draw_cubes = cube is not None and cube > 0 and angles.size == len(positions)
    if draw_cubes:
        unit = np.array([[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
                         [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]], dtype=float) * (cube / 2.0)
        corners = np.tile(unit, (len(positions), 1, 1))
        c, s = np.cos(np.deg2rad(angles)), np.sin(np.deg2rad(angles))
        px = corners[:, :, 0].copy()
        py = corners[:, :, 1].copy()
        corners[:, :, 0] = c[:, None] * px - s[:, None] * py
        corners[:, :, 1] = s[:, None] * px + c[:, None] * py
        corners += positions[:, None, :]
        face_ids = np.array([[0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 5, 4],
                             [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7]])
        polygons = corners[:, face_ids, :].reshape(-1, 4, 3)
        collection = Poly3DCollection(polygons, facecolors=np.repeat(colors, 6),
                                      edgecolors="#334155", linewidths=0.15, alpha=0.27)
        ax.add_collection3d(collection)
    else:
        ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], c=colors, marker="s", s=9, alpha=0.55)
    # 仅对绘图箭头抽样，原始磁块位置及求场数据保持完整。
    show = np.unique(np.linspace(0, len(positions) - 1, min(400, len(positions))).astype(int))
    arrow_length = 0.9 * cube if cube else max(float(np.ptp(positions, axis=0).max()) * 0.022, 1.0)
    ax.quiver(positions[show, 0], positions[show, 1], positions[show, 2],
              vectors[show, 0], vectors[show, 1], vectors[show, 2],
              length=arrow_length, normalize=True, color="#172b4d", linewidth=0.55, arrow_length_ratio=0.3)
    bounds = np.ptp(positions, axis=0) + (cube or 0)
    centers = (positions.min(axis=0) + positions.max(axis=0)) / 2
    limits = np.maximum(bounds * 0.56, 1.0)
    ax.set(xlim=(centers[0] - limits[0], centers[0] + limits[0]),
           ylim=(centers[1] - limits[1], centers[1] + limits[1]),
           zlim=(centers[2] - limits[2], centers[2] + limits[2]),
           xlabel="x [mm]", ylabel="y [mm]", zlabel="z / 轴向 [mm]")
    ax.set_box_aspect(np.maximum(bounds, 1.0))
    ax.view_init(elev=21, azim=-57)
    ax.set_title(_original_label(payload) + "：实际磁块排布与 N 极方向", pad=22)
    ax.scatter([], [], [], color="#dd624e", label="N 极径向朝外")
    ax.scatter([], [], [], color="#3c82b4", label="N 极径向朝内")
    ax.legend(loc="upper left", frameon=True)
    geometry_note = "全部磁块按实际中心及绕 z 轴角度绘制" if draw_cubes else "方点表示实际磁块中心；未绘制磁块实体姿态"
    fig.text(0.5, 0.025, geometry_note + "；磁化箭头显示 %d / %d 个，全部磁块参与计算。" % (len(show), len(positions)),
             ha="center", fontsize=9, color="#475569")
    return _save(fig, output_dir, "01_baseline_geometry.png")


def _plot_axes(output_dir: Path, payload: Mapping[str, Any]) -> Optional[Path]:
    lines = payload.get("plot_data", {}).get("axis_lines", [])
    if not lines:
        return None
    fig, axes = plt.subplots(len(lines), 1, figsize=(10, 3.1 * len(lines)), squeeze=False)
    for ax, line in zip(axes[:, 0], lines):
        coordinate = np.asarray(line["coordinate_mm"], dtype=float)
        field = np.asarray(line["B_T"], dtype=float).reshape(-1, 3)
        ax.plot(coordinate, field[:, 2] * 1000, color="#22649c", linewidth=1.8, label="有符号 Bz")
        norms = np.linalg.norm(field, axis=1) * 1000
        ax.plot(coordinate, norms, color="#dd624e", linestyle="--", linewidth=1.5, label="|B|")
        ax.axvline(0, color="#64748b", linewidth=0.7, linestyle=":")
        if np.nanmin(field[:, 2]) <= 0 <= np.nanmax(field[:, 2]):
            ax.axhline(0, color="#cbd5e1", linewidth=0.7)
        title = "穿过原点的 " + str(line.get("axis", "")).upper() + " 轴线"
        if _is_planar(payload):
            title = "z=" + _fmt(payload["model_info"].get("fov_z_mm", 0)) + " mm 平面内的 " + str(line.get("axis", "")).upper() + " 直径线"
        ax.set(xlabel=str(line.get("axis", "")) + " [mm]", ylabel="磁感应强度 [mT]", title=title)
        ax.grid(True)
        ax.ticklabel_format(style="plain", axis="both", useOffset=False)
        ax.legend(loc="best")
    fig.suptitle("原始物理对照 · " + ("平面 FOV 两条中心直径线" if _is_planar(payload) else "中心轴线") + "的 Bz 与 |B|", fontsize=15, y=1.005)
    fig.tight_layout()
    return _save(fig, output_dir, "02_axis_fields.png")


def _plane_arrays(plane: Mapping[str, Any]) -> Any:
    u = np.asarray(plane["u_mm"], dtype=float)
    v = np.asarray(plane["v_mm"], dtype=float)
    field = np.asarray(plane["B_T"], dtype=float).reshape(len(v), len(u), 3)
    valid = np.asarray(plane.get("mask", np.ones(field.shape[:2], dtype=bool)), dtype=bool).reshape(field.shape[:2]).copy()
    valid &= np.all(np.isfinite(field), axis=-1)
    return u, v, field, valid


def _plot_planes(output_dir: Path, payload: Mapping[str, Any]) -> Optional[Path]:
    planes = payload.get("plot_data", {}).get("planes", [])
    if not planes:
        return None
    arrays = [_plane_arrays(plane) for plane in planes]
    values_bz = [field[:, :, 2][valid] * 1000 for _, _, field, valid in arrays if valid.any()]
    values_norm = [np.linalg.norm(field, axis=-1)[valid] * 1000 for _, _, field, valid in arrays if valid.any()]
    if not values_bz:
        return None
    bz_range = np.concatenate(values_bz)
    norm_range = np.concatenate(values_norm)
    bz_scale = Normalize(float(bz_range.min()), float(bz_range.max()))
    norm_scale = Normalize(float(norm_range.min()), float(norm_range.max()))
    fig, axes = plt.subplots(len(planes), 2, figsize=(12, 4.8 * len(planes)), squeeze=False,
                             layout="constrained")
    for index, (plane, (u, v, field, valid)) in enumerate(zip(planes, arrays)):
        plane_name = str(plane.get("plane", "xy")).lower()
        coordinates = plane_name if len(plane_name) == 2 else "uv"
        plotted = (field[:, :, 2] * 1000, np.linalg.norm(field, axis=-1) * 1000)
        for col, (values, scale, cmap, label) in enumerate(zip(plotted, (bz_scale, norm_scale), ("RdBu_r", "viridis"), ("Bz", "|B|"))):
            ax = axes[index, col]
            im = ax.pcolormesh(u, v, np.ma.array(values, mask=~valid), shading="auto", cmap=cmap, norm=scale)
            ax.set(aspect="equal", xlabel=coordinates[0] + " [mm]", ylabel=coordinates[1] + " [mm]",
                   title=("z=" + _fmt(payload["model_info"].get("fov_z_mm", 0)) + " mm · " if _is_planar(payload) else "") + plane_name.upper() + " / " + label)
            ax.axhline(0, color="#64748b", linewidth=0.4, alpha=0.5)
            ax.axvline(0, color="#64748b", linewidth=0.4, alpha=0.5)
            ax.set_facecolor("#e8edf3")
            if _is_planar(payload):
                radius = float(payload["model_info"]["fov_diameter_mm"])/2
                im.set_clip_path(Circle((0, 0), radius, transform=ax.transData))
                ax.set(xlim=(-radius, radius), ylim=(-radius, radius))
            formatter = ScalarFormatter(useOffset=False)
            formatter.set_scientific(False)
            fig.colorbar(im, ax=ax, shrink=0.9, label=label + " [mT]", format=formatter)
    plane_title = ("z=" + _fmt(payload["model_info"].get("fov_z_mm", 0)) + " mm 平面圆形 FOV" if _is_planar(payload)
                   else str(len(planes)) + " 个过原点截面")
    fig.suptitle("原始物理对照 · " + plane_title + "（同一场量使用相同色标）\n圆外留白为评价域以外，不代表磁场为零", fontsize=14)
    return _save(fig, output_dir, "03_field_sections.png")


def _plot_scans(output_dir: Path, payload: Mapping[str, Any]) -> List[Path]:
    paths = []
    baseline = _scan_reference(payload)
    for index, scan in enumerate(payload.get("scans", []), start=1):
        rows = scan.get("rows", [])
        if not rows:
            continue
        parameter = str(scan.get("parameter", "parameter"))
        parameter_label = str(scan.get("parameter_label") or _parameter_label(parameter))
        value_labels = scan.get("value_labels") or {}
        values = np.array([float(row["value"]) for row in rows])
        valid = np.array([bool(row.get("valid", True)) for row in rows])
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.7))
        for ax, key in zip(axes, ("mean_norm_T", "pp_norm_ppm", "mass_magnet_kg")):
            y = np.array([_metric(row.get("metrics", {}), key) if valid[i] else np.nan for i, row in enumerate(rows)], dtype=float)
            ax.plot(values, y, "o-", color="#22649c", markersize=4, label="本次扫描")
            training_metrics = payload.get("baseline_training_metrics")
            reference = _metric(training_metrics or baseline.get("metrics", {}), key)
            if reference is not None:
                reference_label = "扫描基准（同探索采样）" if training_metrics else "扫描基准密集复核参考"
                ax.axhline(reference, color="#dd624e", linestyle="--", linewidth=1.2, label=reference_label)
                baseline_x = _baseline_parameter(baseline, parameter)
                if baseline_x is not None:
                    ax.scatter([baseline_x], [reference], marker="*", s=150, color="#f7b538", edgecolors="#172b4d", zorder=5, label="扫描基准参数")
            _, unit, _ = METRIC_LABELS[key]
            label = _metric_label(key, payload)
            ax.set(xlabel=parameter_label, ylabel=label + " [" + unit + "]")
            ax.grid(True)
            ax.ticklabel_format(style="plain", axis="both", useOffset=False)
            if value_labels:
                labels = [str(value_labels.get(str(float(value)), value_labels.get(str(value), _fmt(value)))) for value in values]
                ax.set_xticks(values, labels, rotation=25, ha="right", fontsize=8)
        axes[0].legend(fontsize=8)
        fig.suptitle("单参数扫描 · " + parameter_label + "（其他条件按本次扫描记录固定）", fontsize=13)
        if (~valid).any():
            fig.text(0.5, 0.032 if value_labels else 0.005, "有 %d 个参数工况未通过约束或计算检查，未连入有效曲线。" % int((~valid).sum()), ha="center", fontsize=9)
        if value_labels:
            fig.text(0.5, 0.005, "横轴为实际离散选项标签；选项之间的连线仅引导视觉，不表示中间值已计算。", ha="center", fontsize=9)
        fig.tight_layout(rect=(0, 0.06 if value_labels else 0.03, 1, 0.94))
        paths.append(_save(fig, output_dir, "04_scan_%02d.png" % index))
    return paths


def _plot_response(output_dir: Path, payload: Mapping[str, Any]) -> Optional[Path]:
    response = payload.get("response_surface")
    if not response:
        return None
    x = np.asarray(response.get("x_values", []), dtype=float)
    y = np.asarray(response.get("y_values", []), dtype=float)
    if x.size == 0 or y.size == 0:
        return None
    x_name, y_name = str(response["x_name"]), str(response["y_name"])
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.6), layout="constrained")
    baseline = _scan_reference(payload)
    bx, by = _baseline_parameter(baseline, x_name), _baseline_parameter(baseline, y_name)
    for ax, key, cmap in zip(axes, ("mean_norm_T", "pp_norm_ppm"), ("viridis", "magma_r")):
        data = np.asarray(response[key], dtype=float).reshape(len(y), len(x)) * METRIC_LABELS[key][2]
        masked = np.ma.masked_invalid(data)
        im = ax.pcolormesh(x, y, masked, shading="auto", cmap=cmap)
        _, unit, _ = METRIC_LABELS[key]
        label = _metric_label(key, payload)
        fig.colorbar(im, ax=ax, label=label + " [" + unit + "]")
        ax.set(xlabel=response.get("x_label") or _parameter_label(x_name), ylabel=response.get("y_label") or _parameter_label(y_name), title=label)
        if bx is not None and by is not None:
            ax.scatter([bx], [by], marker="*", s=190, color="#f7b538", edgecolors="#172b4d", linewidths=1.0, label="扫描基准")
            ax.legend(loc="best", fontsize=9)
    fig.suptitle("双参数响应面 · 离散计算网格，非连续求解结果", fontsize=14)
    return _save(fig, output_dir, "05_response_surface.png")


def _plot_sensitivity(output_dir: Path, payload: Mapping[str, Any]) -> Optional[Path]:
    data = payload.get("sensitivity", [])
    if isinstance(data, Mapping):
        data = data.get("derivatives", data.get("rows", []))
    columns = (("normalized_strength_sensitivity", "平均 |B| 的局部敏感度"),
               ("normalized_ppm_sensitivity", "峰峰不均匀度的局部敏感度"))
    if not data or not any(any(_number(row.get(key)) is not None for row in data) for key, _ in columns):
        return None
    height = max(5.0, len(data) * 0.34 + 1.6)
    fig, axes = plt.subplots(1, 2, figsize=(14, height))
    for ax, (key, title) in zip(axes, columns):
        rows = [(str(row.get("parameter", row.get("name", ""))), _number(row.get(key))) for row in data]
        rows = [(name, value) for name, value in rows if value is not None]
        rows.sort(key=lambda item: abs(item[1]), reverse=True)
        if not rows:
            ax.set_visible(False)
            continue
        values = np.array([row[1] for row in rows])
        ax.barh(np.arange(len(rows)), values, color=np.where(values >= 0, "#dd624e", "#22649c"), alpha=0.9)
        ax.set_yticks(np.arange(len(rows)), [row[0] for row in rows], fontsize=9)
        ax.invert_yaxis()
        ax.axvline(0, color="#64748b", linewidth=0.8)
        ax.set(title=title, xlabel="导数 × 变量完整范围 / 基准指标（无量纲）")
        ax.grid(True, axis="x")
    fig.suptitle("扫描基准位置的局部有限差分敏感度 · 按绝对值排序\n正负表示指标随参数增加的局部变化方向，不代表全局敏感度", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.92), w_pad=3.0)
    return _save(fig, output_dir, "07_local_sensitivity.png")


def _plot_radial_packing(output_dir: Path, payload: Mapping[str, Any]) -> Optional[Path]:
    manufacturing = payload.get("manufacturing", {})
    config = manufacturing.get("config", {})
    shared = config.get("layout_mode", "independent_pockets") == "radial_channels"
    seed = manufacturing.get("seed_record", {})
    original_design = payload.get("baseline", {}).get("design", {})
    seed_design = seed.get("design", {})
    cube = _number(original_design.get("cube_mm"))
    clearance = _number(config.get("pocket_clearance_per_side_mm"))
    wall = _number(config.get("pocket_web_mm"))
    packing = manufacturing.get("seed_audit", {}).get("packing", {})
    bore = _number(packing.get("protected_assembly_bore_radius_mm"))
    outer = _number(packing.get("outer_assembly_radius_mm"))
    if not seed_design or None in (cube, clearance, wall, bore, outer):
        return None
    layer_index = 1  # 比较本次设计实际的 R02 层。
    original = sorted([row for row in original_design.get("rings", []) if int(row["layer_index"]) == layer_index], key=lambda row: float(row["radius_mm"]))
    printable = sorted([row for row in seed_design.get("rings", []) if int(row["layer_index"]) == layer_index], key=lambda row: float(row["radius_mm"]))
    if not original or not printable:
        return None
    pocket = cube + 2 * clearance
    required_pitch = _number(packing.get("required_radial_center_pitch_mm"))
    required_gap = max(0., required_pitch - cube) if required_pitch is not None else (0. if shared else pocket + wall - cube)
    fig, ax = plt.subplots(figsize=(13, 5.4))
    for center_y, rows in ((1.2, original), (0.0, printable)):
        radii = np.array([float(row["radius_mm"]) for row in rows])
        if shared:
            # One continuous pocket for the column: end clearance is not
            # replicated between individual magnets in this channel.
            channel_start, channel_end = radii[0]-pocket/2, radii[-1]+pocket/2
            ax.add_patch(Rectangle((channel_start, center_y-.29), channel_end-channel_start, .58,
                                   facecolor="#dbeafe", edgecolor="#3b82b6", linewidth=1.2))
        for radius in radii:
            if not shared:
                ax.add_patch(Rectangle((radius-pocket/2, center_y-.29), pocket, .58,
                                       facecolor="#dbeafe", edgecolor="#3b82b6", linewidth=1.2))
            ax.add_patch(Rectangle((radius-cube/2, center_y-.2), cube, .4,
                                   facecolor="#8d9fb2", edgecolor="#334155", linewidth=.7))
            ax.text(radius, center_y+.39, _fmt(radius-cube/2, 3), ha="center", va="bottom", fontsize=9)
        if len(radii) > 1:
            if shared:
                gap = float(np.min(np.diff(radii))-cube)
                requirement = required_gap
                description = "实际 %d 个径向环；最小实体面隙 %s mm，要求 ≥ %s mm（允许贴紧）" % (len(radii), _fmt(gap,3), _fmt(requirement))
            else:
                gap = float(np.min(np.diff(radii))-pocket)
                requirement = wall
                description = "实际 %d 个径向环；最小槽间净距 %s mm，要求 ≥ %s mm" % (len(radii), _fmt(gap,3), _fmt(requirement))
            ax.text((bore+outer)/2, center_y-.48, description, ha="center", va="top",
                    color="#ba4c37" if gap < requirement-1e-7 else "#247a60", fontsize=10)
    ax.axvline(bore, color="#172b4d", linestyle="--", linewidth=1.2)
    ax.axvline(outer, color="#172b4d", linestyle="--", linewidth=1.2)
    ax.text(bore,1.91,"保护孔 r="+_fmt(bore)+" mm",ha="left",va="bottom",fontsize=10)
    ax.text(outer,1.91,"外包络 R="+_fmt(outer,3)+" mm",ha="right",va="bottom",fontsize=10)
    ax.set(xlim=(bore-8,outer+8),ylim=(-.8,2.1),xlabel="径向位置 R [mm]；磁块数字为内侧底边半径",
           title=("同列连续径向槽：磁铁允许贴紧，端部保留内外壁余量" if shared else "独立磁块槽口的径向排布：原始与打印起点"))
    ax.set_yticks([1.2,0.],["R02 原始","R02 起点"])
    ax.ticklabel_format(style="plain",axis="x",useOffset=False)
    ax.grid(True,axis="x")
    ax.legend(handles=[Patch(facecolor="#8d9fb2",edgecolor="#334155",label="真实磁块："+_fmt(cube)+" mm"),
                       Patch(facecolor="#dbeafe",edgecolor="#3b82b6",label=("整列连续槽，端部余量："+_fmt(clearance)+" mm/侧" if shared else "单块槽包络："+_fmt(pocket)+" mm"))],
              loc="upper center",bbox_to_anchor=(.5,-.16),ncol=2,frameon=False)
    note = ("同一径向列不要求块间隔墙；最内/外端保留槽端余量和圆环侧壁。不同周向列仍检查列间隔墙，实体禁止重叠。" if shared else "隔墙要求来自每块磁铁独立槽的历史工艺假设；不代表连续槽也要求块间隔墙。")
    fig.text(.5,.01,"数字为内侧底边半径 R底=R中心−边长/2；纵向仅分栏。"+note,ha="center",fontsize=8.5,color="#475569")
    fig.tight_layout(rect=(0,.08,1,1))
    return _save(fig,output_dir,"08_print_radial_packing.png")


def _candidate_rows(payload: Mapping[str, Any]) -> List[Mapping[str, Any]]:
    candidates = [dict(row) for row in payload.get("candidates", [])]
    baseline = payload.get("baseline", {})
    baseline_id = str(baseline.get("candidate_id", "baseline"))
    if baseline and not any(str(row.get("candidate_id")) == baseline_id for row in candidates):
        candidates.insert(0, dict(baseline))
    manufacturing = payload.get("manufacturing", {})
    seed = manufacturing.get("seed_record", {})
    if seed and record_metrics(seed) and not any(str(row.get("candidate_id")) == str(seed.get("candidate_id", "printable_seed")) for row in candidates):
        seed = dict(seed)
        seed.setdefault("candidate_id", "printable_seed")
        candidates.insert(1, seed)
    if manufacturing:
        for row in candidates:
            row["_is_manufacturing_seed"] = bool(seed) and str(row.get("candidate_id")) == str(seed.get("candidate_id", "printable_seed"))
            audit = _manufacturing_audit(row, payload)
            row["_manufacturing_review_required"] = True
            if audit is not None:
                row["manufacturing_check"] = audit
                if not audit.get("valid", False):
                    row["feasible"] = False
    for row in candidates:
        row['metrics'] = record_metrics(row)
        if payload.get('requirements') is not None:
            review = (check_performance(row['metrics'],payload['requirements']) if valid_record_metrics(row)
                      else {'valid':False,'violations':['invalid field metrics']})
            if not review['valid']:
                row['performance_check'] = review
                row['validation_status'] = {'label':'独立复核未通过','dense_pass':False}
        if row.get("performance_check", {}).get("valid") is False:
            row["feasible"] = False
    return candidates


def _pareto_flags(candidates: Sequence[Mapping[str, Any]], requirements=None) -> np.ndarray:
    selected = {id(row) for row in pareto_front(eligible_records(candidates,requirements), requirements)}
    return np.asarray([id(row) in selected for row in candidates], dtype=bool)


def _plot_pareto(output_dir: Path, payload: Mapping[str, Any], candidates: Sequence[Mapping[str, Any]]) -> Optional[Path]:
    if len(candidates) < 2:
        return None
    baseline_id = str(payload.get("baseline", {}).get("candidate_id", "baseline"))
    rows = []
    for index, row in enumerate(candidates):
        m = record_metrics(row)
        mass, ppm, field = (_number(m.get(key)) for key in ("mass_magnet_kg", "pp_norm_ppm", "mean_norm_T"))
        if None not in (mass, ppm, field):
            rows.append((index, row, mass, ppm, field * 1000))
    if not rows:
        return None
    pareto = _pareto_flags(candidates, payload.get('requirements'))
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.8), layout="constrained")
    masses = np.array([r[2] for r in rows])
    ppms = np.array([r[3] for r in rows])
    fields = np.array([r[4] for r in rows])
    field_label = _metric_label("mean_norm_T", payload) + " [mT]"
    for ax, x, xlabel in zip(axes, (masses, fields), ("永磁材料质量 [kg]", field_label)):
        im = ax.scatter(x, ppms, c=fields, cmap="viridis", s=32, alpha=0.65, label="真实场计算结果")
        selected = np.array([pareto[r[0]] for r in rows])
        if selected.any():
            front_label = "几何与性能通过的非支配候选" if payload.get("requirements") else "几何初筛通过的双目标非支配"
            ax.scatter(x[selected], ppms[selected], facecolors="none", edgecolors="#172b4d", s=85, linewidths=0.85, label=front_label)
        for point, row in enumerate(rows):
            if str(row[1].get("candidate_id")) == baseline_id:
                ax.scatter([x[point]], [ppms[point]], marker="*", color="#f7b538", edgecolors="#172b4d", s=240, zorder=8, label=_original_label(payload))
                ax.annotate("baseline", (x[point], ppms[point]), xytext=(7, 8), textcoords="offset points", fontsize=9, weight="bold")
            elif str(row[1].get("candidate_id")) == str(payload.get("manufacturing", {}).get("seed_record", {}).get("candidate_id", "printable_seed")):
                seed_label = "打印起点（几何初筛通过）" if row[1].get("manufacturing_check", {}).get("valid") else "打印起点（尚待初筛）"
                if row[1].get("performance_check", {}).get("valid") is False:
                    seed_label = "打印起点（性能未达标）"
                ax.scatter([x[point]], [ppms[point]], marker="D", color="#38a68b", edgecolors="#172b4d", s=100, zorder=7, label=seed_label)
                ax.annotate(str(row[1].get("candidate_id", "")), (x[point], ppms[point]), xytext=(7, -14), textcoords="offset points", fontsize=8)
        ax.set(xlabel=xlabel, ylabel="|B| 峰峰不均匀度 [ppm]")
        ax.grid(True)
        maximum_ppm = _number(payload.get("requirements", {}).get("max_pp_norm_ppm"))
        if maximum_ppm is not None:
            ax.axhline(maximum_ppm, color="#b85245", linestyle="--", linewidth=1, label="ppm 上限")
    minimum_field = _number(payload.get("requirements", {}).get("min_mean_norm_T"))
    if minimum_field is not None:
        axes[1].axvline(minimum_field * 1000, color="#b85245", linestyle="--", linewidth=1, label="平均场强下限")
        axes[1].legend(fontsize=8, loc="best")
    axes[0].legend(fontsize=8, loc="best")
    fig.colorbar(im, ax=axes, shrink=0.85, label=field_label)
    fig.suptitle("双目标候选比较 · 场强与均匀性（质量为诊断指标）\n非支配关系仅针对本次有效候选，不代表全局最优", fontsize=13)
    return _save(fig, output_dir, "06_pareto_candidates.png")


def _metric_table(metrics: Mapping[str, Any], payload: Optional[Mapping[str, Any]] = None) -> str:
    rows = []
    for key, (label, unit, _) in METRIC_LABELS.items():
        if key in metrics:
            rows.append("<tr><th>" + _esc(_metric_label(key, payload)) + "</th><td>" + _fmt(_metric(metrics, key)) + "</td><td>" + _esc(unit) + "</td></tr>")
    return "<table><thead><tr><th>指标</th><th>计算结果</th><th>单位</th></tr></thead><tbody>" + "".join(rows) + "</tbody></table>"


def _cached_line_table(model_info: Mapping[str, Any]) -> str:
    data = model_info.get("cached_line_comparison")
    if not data:
        return ""
    rows = []
    for label, python_key, fem_key, bias_key in (
        ("Bz 最小值", "python_min_mT", "comsol_saved_min_mT", "minimum_bias_percent"),
        ("Bz 最大值", "python_max_mT", "comsol_saved_max_mT", "maximum_bias_percent"),
    ):
        rows.append("<tr><th>" + label + "</th><td>" + _fmt(data.get(python_key), 8) + "</td><td>" + _fmt(data.get(fem_key), 8) + "</td><td>" + _fmt(data.get(bias_key), 4) + "%</td></tr>")
    planar = model_info.get("fov_geometry") == "disk"
    scope = ("仅对照原模型 x=-15 至 15 mm、y=z=0 的历史截线极值，不代表本次平面 FOV 指标"
             if planar else data.get("scope", "仅对照原模型已保存的截线极值"))
    content = "<h3>原始磁体已有 30 mm 截线结果的有限对照</h3><p>" + _esc(scope) + "。</p>"
    content += "<table><thead><tr><th>截线指标</th><th>Python [mT]</th><th>COMSOL 已保存值 [mT]</th><th>相对偏差</th></tr></thead><tbody>" + "".join(rows) + "</tbody></table>"
    if planar:
        content += "<p class='caption'>偏差 = (Python−COMSOL)/COMSOL×100%。历史截线长度30 mm，与本次φ" + _fmt(model_info.get("fov_diameter_mm")) + " mm平面FOV范围不同；这里只作历史参考，未据此计算FOV面积平均场或均匀性。本次平面同坐标FEM校准尚未完成。</p>"
    else:
        content += "<p class='caption'>偏差 = (Python−COMSOL)/COMSOL×100%。这是原始磁体已有一维极值表的对照，尚未完成本次 DSV 同坐标三维 FEM 校准；打印起点与候选尚未完成 FEM 对标。未用两端极值推导缺失的平均值或 DSV 均匀性。</p>"
    if data.get("point_export_status"):
        content += "<p class='caption'>同坐标场数据导出状态：" + _esc(data["point_export_status"]) + "。原始 .mph 未修改。</p>"
    return content


def _image_section(path: Path, title: str, caption: str = "") -> str:
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return '<section><h2>' + _esc(title) + '</h2><img src="data:image/png;base64,' + encoded + '" alt="' + _esc(title) + '">' + ('<p class="caption">' + _esc(caption) + '</p>' if caption else '') + '</section>'


def _candidate_table(candidates: Sequence[Mapping[str, Any]], baseline_id: str, payload: Optional[Mapping[str, Any]] = None) -> str:
    if len(candidates) <= 1:
        return "<p>本次未执行候选方案搜索。</p>"
    pareto = _pareto_flags(candidates, (payload or {}).get('requirements'))
    chosen = list(enumerate(candidates))
    chosen.sort(key=lambda item: (str(item[1].get("candidate_id")) != baseline_id,
                                  _number(item[1].get("metrics", {}).get("pp_norm_ppm")) or 0))
    rows = []
    for i, row in chosen[:30]:
        metrics = record_metrics(row)
        baseline = str(row.get("candidate_id")) == baseline_id
        deliveries={r.get('id',r.get('candidate_id')):r for r in (payload or {}).get('deliverables',{}).get('candidates',[])}
        export_item=deliveries.get(row.get('candidate_id'),{})
        name = _esc('原始裸磁体' if baseline else 'PAVA打印起点' if row.get('_is_manufacturing_seed') else export_item.get('display_name','候选'))
        status = "原始物理对照" if baseline else ("打印起点" if row.get("_is_manufacturing_seed") else ("本次非支配" if pareto[i] else "候选"))
        geometry_status = _manufacturing_label(row.get("manufacturing_check")) if row.get("_manufacturing_review_required") else ("几何通过" if row.get("geometry_check", {}).get("valid") else "几何未通过")
        performance = row.get("performance_check", {})
        performance_status = ("性能门槛通过" if performance.get("valid") else "性能未达标") if performance else "历史结果未设置门槛"
        if performance.get("valid") is False:
            status += " / 仅作诊断"
        violations = performance.get("violations", [])
        if isinstance(violations, str):
            violations = [violations]
        if violations:
            performance_status += "：" + "; ".join(_violation_text(v) for v in violations)
        detail = json.dumps(_json_safe({'candidate_id':row.get('candidate_id'),'parameters':row.get('parameters',{}),'validation_status':row.get('validation_status')}), ensure_ascii=False)
        performance_status=validation_label(row) + ('：'+ '; '.join(_violation_text(v) for v in violations) if violations else '')
        training=row.get('training_performance_check') or {}
        training_label='通过' if training.get('valid') else '未通过' if training else '未记录' 
        cells = [name, _esc(status), _esc(geometry_status), _esc(performance_status), _fmt(_metric(metrics, "mean_norm_T")), _fmt(_metric(metrics, "pp_norm_ppm")),
                 _fmt(_metric(metrics, "mass_magnet_kg")), "<details><summary>参数与溯源</summary><pre>" + _esc(detail) + "</pre></details>", _esc(training_label)]
        rows.append(('<tr class="baseline">' if baseline else '<tr>') + "".join("<td>" + cell + "</td>" for cell in cells) + "</tr>")
    note = "" if len(chosen) <= 30 else "<p class='caption'>表中展示基准及按不均匀度排序的前 29 个已复核方案，其余见候选数据文件。</p>"
    return "<div class='table-wrap'><table><thead><tr><th>方案</th><th>标记</th><th>几何检查</th><th>性能门槛</th><th>" + _esc(_metric_label("mean_norm_T", payload)) + " [mT]</th><th>不均匀度 [ppm]</th><th>磁材 [kg]</th><th>内部参数（折叠）</th><th>训练门槛</th></tr></thead><tbody>" + "".join(rows) + "</tbody></table></div>" + note


def _violation_text(value: Any) -> str:
    if isinstance(value, Mapping):
        for key in ("message", "description", "reason"):
            if key in value:
                return str(value[key])
        return json.dumps(_json_safe(value), ensure_ascii=False)
    return str(value)


def _audit_card(title: str, audit: Optional[Mapping[str, Any]]) -> str:
    if not audit:
        return "<h3>" + _esc(title) + "</h3><p>尚未提供制造审查结果。</p>"
    status = _manufacturing_label(audit)
    violations = audit.get("violations", [])
    if isinstance(violations, (str, Mapping)):
        violations = [violations]
    result = "<h3>" + _esc(title) + "</h3><p><strong>" + _esc(status) + "</strong>。"
    if audit.get("valid") is False:
        result += "这表示未通过当前制造几何筛选，不表示其磁场物理模型无效。"
    result += "</p>"
    checks = audit.get("checks", {})
    if checks:
        labels = {
            "nominal_magnet_geometry": "磁块名义几何", "magnet_budget": "磁块数量预算",
            "unique_slot_ids": "磁块槽位编号唯一", "shared_azimuth_columns": "同层各环周向列对齐",
            "columns_divisible_by_sectors": "周向列可按扇区整分", "shared_column_phase": "同层各环起始角一致",
            "pocket_inner_wall": "内壁余量", "pocket_outer_wall": "外壁余量",
            "necessary_radial_packing_bound": "径向槽口和隔墙容纳数量", "module_axial_envelope": "模块轴向外包络",
            "module_axial_separation": "层间模块净间隙", "pocket_web_obb_clearance": "独立槽之间隔墙净距",
            "sector_fits_printer_bed": "分扇零件的打印床容纳",
        }
        mode = audit.get("process_assumptions", {}).get("layout_mode", audit.get("packing", {}).get("layout_mode", "independent_pockets"))
        if mode == "radial_channels":
            labels["necessary_radial_packing_bound"] = "端壁内可容纳径向磁铁数量"
            labels["pocket_web_obb_clearance"] = "不同周向列之间隔墙净距（同列可贴紧）"
        result += "<table><thead><tr><th>初筛项目</th><th>结果</th></tr></thead><tbody>"
        for key, passed in checks.items():
            result += "<tr><th>" + _esc(labels.get(key, key)) + "</th><td>" + ("通过" if passed else "未通过") + "</td></tr>"
        result += "</tbody></table>"
    if violations:
        result += "<details><summary>展开具体违规记录（保留原检查信息）</summary><pre>" + _esc(json.dumps(_json_safe(violations), ensure_ascii=False, indent=2)) + "</pre></details>"
    return result


def _layer_summary_table(baseline_audit: Mapping[str, Any], seed_audit: Mapping[str, Any]) -> str:
    original = {int(row["layer_index"]): row for row in baseline_audit.get("layer_modules", []) if "layer_index" in row}
    printable = {int(row["layer_index"]): row for row in seed_audit.get("layer_modules", []) if "layer_index" in row}
    if not original and not printable:
        return ""
    rows = []
    for index in sorted(set(original) | set(printable)):
        a, b = original.get(index, {}), printable.get(index, {})
        def columns(record: Mapping[str, Any]) -> str:
            value = record.get("azimuth_columns", record.get("magnets_per_ring"))
            if isinstance(value, list):
                return "/".join(_fmt(item) for item in sorted(set(value)))
            return _fmt(value)
        cells = ["R%02d" % (index + 1), _fmt(a.get("radial_ring_count")), _fmt(b.get("radial_ring_count")),
                 columns(a), columns(b), _fmt(a.get("magnet_count")), _fmt(b.get("magnet_count")),
                 _fmt(b.get("module_thickness_mm"))]
        rows.append("<tr>" + "".join("<td>" + _esc(cell) + "</td>" for cell in cells) + "</tr>")
    return "<h3>逐层数量与打印模块</h3><div class='table-wrap'><table><thead><tr><th>层</th><th>原始径向环</th><th>起点径向环</th><th>原始周向列/环</th><th>起点周向列/环</th><th>原始磁块</th><th>起点磁块</th><th>起点模块厚 [mm]</th></tr></thead><tbody>" + "".join(rows) + "</tbody></table></div>"


def _manufacturing_section(payload: Mapping[str, Any], candidates: Sequence[Mapping[str, Any]]) -> str:
    manufacturing = payload.get("manufacturing", {})
    if not manufacturing:
        return "<section><h2>制造审查状态</h2><div class='notice'>本次属于历史物理预测，未按新增的独立磁块槽口、隔墙与打印床约束验收；不能把候选方案视为可制造推荐。</div></section>"
    config = manufacturing.get("config", {})
    shared = config.get("layout_mode", "independent_pockets") == "radial_channels"
    parameters = [
        ("槽端/侧单边余量" if shared else "槽口单边余量", "pocket_clearance_per_side_mm", "mm"),
        ("不同周向列之间最小隔墙" if shared else "独立槽之间最小隔墙", "pocket_web_mm", "mm"),
        ("内壁", "inner_wall_mm", "mm"),
        ("外壁", "outer_wall_mm", "mm"),
        ("底板厚度", "base_thickness_mm", "mm"),
        ("每层打印扇区", "sectors_per_layer", "个"),
        ("打印床尺寸", "printer_bed_mm", "mm"),
        ("床面每侧预留", "bed_margin_per_side_mm", "mm"),
        ("层间模块最小净距", "min_axial_module_gap_mm", "mm"),
        ("磁块数量上限", "max_magnet_count", "块"),
    ]
    rows = []
    for label, key, unit in parameters:
        if key not in config:
            continue
        value = config[key]
        text = " × ".join(_fmt(item) for item in value) if isinstance(value, (list, tuple)) else _fmt(value)
        rows.append("<tr><th>" + _esc(label) + "</th><td>" + _esc(text) + "</td><td>" + unit + "</td></tr>")
    original_audit, seed_audit = manufacturing.get("baseline_audit", {}), manufacturing.get("seed_audit", {})
    mode_label = "同列连续径向槽" if shared else "每块磁铁独立槽口（历史/兼容模式）"
    result = "<section><h2>制造约束 · " + mode_label + "</h2>"
    explanation = ("同一径向列的磁铁可以贴紧，不施加块间槽隙或隔墙；最内和最外磁铁与圆环侧壁保留端部余量。不同周向列继续检查列间隔墙，所有磁块仍禁止实体重叠。"
                   if shared else "本记录采用每块磁铁独立定位槽的历史工艺假设，因而要求槽间隔墙；不能用此要求解释新的连续径向槽方案。")
    result += "<div class='notice'>" + explanation + "余量、壁厚和打印床参数均为本次工艺值；几何初筛不替代支架CAD或强度验证。</div>"
    result += "<table><thead><tr><th>工艺参数</th><th>本次值</th><th>单位</th></tr></thead><tbody>" + "".join(rows) + "</tbody></table>"
    result += _audit_card("原始物理对照", original_audit) + _audit_card("打印起点（非优化解）", seed_audit)
    result += "<h3>原始、打印起点与候选的物理结果</h3><div class='table-wrap'><table><thead><tr><th>方案</th><th>磁块</th><th>径向环总数</th><th>平均 |B| [mT]</th><th>不均匀度 [ppm]</th><th>磁材 [kg]</th><th>制造状态</th></tr></thead><tbody>"
    for row in candidates:
        metrics = row.get("metrics", {})
        rings = row.get("design", {}).get("rings", [])
        audit = _manufacturing_audit(row, payload)
        cells = [str(row.get("candidate_id", "")), _fmt(metrics.get("magnet_count")), _fmt(len(rings)) if rings else "—",
                 _fmt(_metric(metrics, "mean_norm_T")), _fmt(_metric(metrics, "pp_norm_ppm")),
                 _fmt(_metric(metrics, "mass_magnet_kg")), _manufacturing_label(audit)]
        result += "<tr>" + "".join("<td>" + _esc(cell) + "</td>" for cell in cells) + "</tr>"
    result += "</tbody></table></div><p class='caption'>磁材质量仅作诊断。原始对照即使物理指标良好，也不进入本轮制造可行 Pareto 集；打印起点通过制造几何检查不等于已找到最优方案。</p>"
    quantity_rows = manufacturing.get("quantity_tradeoffs", [])
    if quantity_rows:
        result += "<h3>固定径向环布局的数量试算</h3><div class='table-wrap'><table><thead><tr><th>每环周向列</th><th>径向环总数</th><th>磁块总数</th><th>数量预算</th><th>平均 |B| [mT]</th><th>不均匀度 [ppm]</th><th>初筛状态</th></tr></thead><tbody>"
        for row in quantity_rows:
            count, budget = _number(row.get("magnet_count")), _number(row.get("magnet_count_budget"))
            status = "几何工艺初筛通过" if row.get("manufacturing_valid") else ("超出磁块数量预算" if count is not None and budget is not None and count > budget else "工艺初筛未通过")
            row_metrics = row.get("metrics", {})
            cells = [_fmt(row.get("azimuth_columns")), _fmt(row.get("radial_rings_total")), _fmt(count), _fmt(budget),
                     _fmt(_metric(row_metrics, "mean_norm_T")), _fmt(_metric(row_metrics, "pp_norm_ppm")), status]
            result += "<tr>" + "".join("<td>" + _esc(cell) + "</td>" for cell in cells) + "</tr>"
        result += "</tbody></table></div><p class='caption'>本表固定径向环布局，只改变周向列；物理值来自探索采样，不是密集复核。— 表示该工况未求场，不表示磁场为零。预算不通过不等于所有同列数排布均不可行，还需结合径向环子集一起判断。</p>"
    result += _layer_summary_table(original_audit, seed_audit)
    result += "<details><summary>展开本次制造配置与工艺假设</summary><pre>" + _esc(json.dumps(_json_safe({"config": config, "process_assumptions": seed_audit.get("process_assumptions", {})}), ensure_ascii=False, indent=2)) + "</pre></details></section>"
    return result


def _performance_section(payload: Mapping[str, Any], candidates: Sequence[Mapping[str, Any]]) -> str:
    requirements = payload.get("requirements", {})
    summary = payload.get("performance_summary", {})
    if not requirements and not summary:
        return ""
    met = bool(summary.get("requirements_met", False))
    heading = "已找到通过当前门槛的候选" if met else "本次尚未找到达标候选"
    content = "<section><h2>" + heading + "</h2>"
    if met:
        content += "<p>至少一个方案同时通过本轮几何检查和独立复核后的性能门槛。它仍是快速解析候选，FEM 与制造验证需另行完成。</p>"
    else:
        content += "<div class='notice'><strong>本次未找到同时满足几何条件和性能门槛的方案。</strong>下方保留的未达标方案及其布局／CLASS仅供诊断，不能作为达标最优解或制造定案。</div>"
    minimum = _number(requirements.get("min_mean_norm_T"))
    maximum = _number(requirements.get("max_pp_norm_ppm"))
    tolerance = _number(requirements.get("field_tolerance_T"))
    rows = [
        (_metric_label("mean_norm_T", payload) + " 下限", _fmt(minimum*1000) + " mT" if minimum is not None else "未设置"),
        ("场强参考", str(requirements.get("reference", "显式输入"))),
        ("场强判定容差", _fmt(tolerance*1000) + " mT" if tolerance is not None else "0 mT"),
        ("峰峰值 ppm 上限", _fmt(maximum) if maximum is not None else "未设置；达到场强门槛后优先降低 ppm"),
        ("密集复核后可行帕累托候选", ", ".join(str(v) for v in summary.get("verified_pareto_ids", [])) or "无"),
    ]
    content += "<table><tbody>" + "".join("<tr><th>"+_esc(k)+"</th><td>"+_esc(v)+"</td></tr>" for k,v in rows) + "</tbody></table>"
    content += "<p class='caption'>几何通过与性能达标分别判定；搜索使用可行性优先的 ε 约束，当前门槛内首先比较 ppm。未设置 ppm 上限不等于 ppm 已满足成像要求。</p>"
    return content + "</section>"


def _deliverables_section(output_dir: Path, payload: Mapping[str, Any]) -> str:
    deliveries = payload.get("deliverables", {})
    if not deliveries:
        return ""
    rows = deliveries.get("candidates", [])
    result = "<section><h2>候选磁铁布局与 COMSOL 模型</h2>"
    result += "<p>布局图和模型文件读取同一候选排布。CLASS 文件构建几何、磁化与物理场设置；编译成功表示文件可供 COMSOL 加载，FEM 求解与精度验证需另行执行。</p>"
    if not rows:
        result += "<p>" + ("parameters.py 已关闭自动导出。" if deliveries.get("enabled") is False else "本次没有可导出的制造可行候选。") + "</p>"
    for row in rows:
        ident = row.get("id", row.get("candidate_id", "候选"))
        compiled = bool(row.get("compiled"))
        label = str(row.get("display_name") or ident) + ("（打印起点）" if ident == "manufacturing_seed" else "")
        if row.get("diagnostic_only"):
            label += "（未达标，仅供诊断）"
        result += "<h3>" + _esc(label) + "</h3>"
        if row.get("diagnostic_only"):
            result += "<div class='notice'>该方案未满足本轮性能门槛。布局图和CLASS用于分析原因，编译成功不改变其未达标状态。</div>"
        result += "<p><strong>" + ("CLASS 编译成功" if compiled else "CLASS 尚未成功编译") + "</strong>"
        if row.get("status"):
            result += " · " + _esc(row["status"])
        result += "</p>"
        if row.get("error"):
            result += "<div class='notice'>导出错误：" + _esc(row["error"]) + "</div>"
        links = []
        for key, label in (("class_file", "下载 CLASS"), ("java_file", "查看 Java 源文件"), ("layout_file", "打开布局 PNG")):
            if not row.get(key):
                continue
            path = Path(row[key])
            if not path.is_absolute():
                path = output_dir / path
            if not path.is_file():
                continue
            relative = os.path.relpath(str(path), str(output_dir)).replace(os.sep, "/")
            links.append('<a href="' + _esc(quote(relative, safe="/")) + '">' + label + '</a>')
        if links:
            result += "<p>" + " · ".join(links) + "</p>"
        if row.get("layout_file"):
            path = Path(row["layout_file"])
            if not path.is_absolute():
                path = output_dir / path
            if path.is_file():
                encoded = base64.b64encode(path.read_bytes()).decode("ascii")
                result += '<img src="data:image/png;base64,' + encoded + '" alt="' + _esc(ident) + ' R-Z 磁铁布局">'
    return result + "</section>"



def _local_media(output_dir, value):
    if not value:return None
    path=Path(value)
    if not path.is_absolute():path=output_dir/path
    return path if path.is_file() else None


def _hero_section(output_dir, payload):
    """Keep the actual selected R-Z arrangement as the report's main visual."""
    chosen=best_record(payload)
    exports=payload.get('deliverables',{}).get('candidates',[])
    item=next((r for r in exports if r.get('candidate_id',r.get('id'))==chosen.get('candidate_id')),None)
    if item is None:item=next((r for r in exports if not r.get('diagnostic_only')),exports[0] if exports else {})
    result='<section class="layout-hero"><h2>核心结果 · 磁铁布局与起点位移</h2>'
    result+='<p><strong>'+_esc(validation_label(chosen))+'</strong>。彩色实心块为当前排布；灰色虚线框为PAVA打印起点。ΔR、ΔZ均为物理毫米变化，精确坐标见运行速览与JSON。</p>'
    path=_local_media(output_dir,(item or {}).get('layout_file'))
    if path:
        encoded=base64.b64encode(path.read_bytes()).decode('ascii')
        result+='<img src="data:image/png;base64,'+encoded+'" alt="本轮选中磁铁布局与PAVA起点比较">'
    else:result+='<p class="caption">本次未生成候选布局图；可用export命令按保存的实际设计补导出。</p>'
    links=[]
    for key,label in [('class_file','COMSOL CLASS'),('java_file','Java源文件'),('layout_file','独立布局PNG')]:
        file=_local_media(output_dir,(item or {}).get(key))
        if file:
            rel=os.path.relpath(str(file),str(output_dir)).replace(os.sep,'/')
            links.append('<a href="'+_esc(quote(rel,safe='/'))+'">'+label+'</a>')
    overview=_local_media(output_dir,payload.get('overview_file'))
    if overview:links.insert(0,'<a href="'+_esc(quote(str(payload['overview_file']).replace('\\','/'),safe='/'))+'">本次运行速览</a>')
    if links:result+='<p>'+' · '.join(links)+'</p>'
    return result+'</section>'


def _evolution_section(output_dir,payload):
    evolution=payload.get('evolution') or {}
    if not evolution.get('enabled'):return ''
    result='<section><h2>磁铁布局与平面FOV · 演化回放</h2>'
    result+='<p class="caption">只在已记录的改善或阶段事件刷新。动画是优化迭代快照，不是磁场随物理时间变化。图中FOV始终为本次固定z平面的圆盘。</p>'
    gif=_local_media(output_dir,evolution.get('gif_file'))
    preview=_local_media(output_dir,evolution.get('preview_file'))
    file=gif or preview
    if file:
        mime='image/gif' if file.suffix.lower()=='.gif' else 'image/png'
        result+='<img src="data:'+mime+';base64,'+base64.b64encode(file.read_bytes()).decode('ascii')+'" alt="磁铁布局及平面磁场演化">'
    else:result+='<p>本次没有可显示的演化帧，数值结果仍按独立复核记录判断。</p>'
    timeline=_local_media(output_dir,evolution.get('timeline_file'))
    if timeline:
        rel=os.path.relpath(str(timeline),str(output_dir)).replace(os.sep,'/')
        result+='<p><a href="'+_esc(quote(rel,safe='/'))+'">打开可逐帧审查的HTML时间轴</a></p>'
    result+='<details><summary>演化记录与绘图开销</summary><pre>'+_esc(json.dumps(_json_safe(evolution),ensure_ascii=False,indent=2))+'</pre></details>'
    return result+'</section>'

def create_report(output_dir: Any, payload: Mapping[str, Any], details_dir: Any = None) -> Dict[str, Any]:
    """从真实结果生成独立 HTML；可把图表和 JSON 收纳到 details_dir。"""
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    details_dir = Path(details_dir).resolve() if details_dir is not None else output_dir
    details_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = details_dir / "图表" if details_dir != output_dir else output_dir
    figures_dir.mkdir(parents=True, exist_ok=True)
    _configure_style()
    baseline = payload.get("baseline", {})
    metrics = baseline.get("metrics", {})
    model_info = payload.get("model_info", {})
    planar = _is_planar(payload)
    if planar:
        evaluation_label = "z=" + _fmt(model_info.get("fov_z_mm", 0)) + " mm 平面圆形 FOV · φ" + _fmt(model_info.get("fov_diameter_mm")) + " mm"
        cylinder_label = ""
    else:
        evaluation_label = "DSV " + _fmt(model_info.get("dsv_diameter_mm")) + " mm 球体" if _number(model_info.get("dsv_diameter_mm")) is not None else "本次配置的球形 DSV"
        cylinder_label = "φ" + _fmt(model_info.get("cylinder_diameter_mm")) + " × " + _fmt(model_info.get("cylinder_length_mm")) + " mm 圆柱" if _number(model_info.get("cylinder_diameter_mm")) is not None else "本次配置的辅助圆柱"
    figures = {}
    sections = []
    figure_tasks = [
        ("geometry", _plot_geometry, "磁体几何与磁化方向", "图中几何取自本次基准方案；坐标单位为 mm。"),
        ("axis_fields", _plot_axes, "平面 FOV 中心直径线" if planar else "轴线磁场", "Bz 保留正负号；|B| 为三分量合成模长。纵轴按实际数据范围放大。" + ("X、Y 两条直径线均位于本次固定 z 平面，曲线重合不代表整个圆盘均匀。" if planar else "两条曲线重合时不表示整个三维区域均匀。")),
        ("field_sections", _plot_planes, "平面 FOV 的 XY 磁场分布" if planar else "XY / XZ / YZ 场分布", "绘图使用真实离散采样数据；圆形掩膜外不参与该截面评价。"),
    ]
    for key, plot, title, caption in figure_tasks:
        path = plot(figures_dir, payload)
        if path:
            figures[key] = str(path)
            sections.append(_image_section(path, title, caption))
    for index, path in enumerate(_plot_scans(figures_dir, payload), start=1):
        figures["scan_%02d" % index] = str(path)
        scan_caption = "扫描与基准虚线使用相同探索采样；主表的密集复核指标另行报告。星标按明确基准参数定位。" if payload.get("baseline_training_metrics") else "扫描使用探索采样；基准虚线来自独立密集采样和极值细化，只作参考。候选需在共同密集网格复核后比较。星标仅按明确基准参数定位。"
        sections.append(_image_section(path, "单参数扫描 %d" % index, scan_caption))
    sensitivity = _plot_sensitivity(figures_dir, payload)
    if sensitivity:
        figures["local_sensitivity"] = str(sensitivity)
        sections.append(_image_section(sensitivity, "归一化局部敏感度", "扫描基准位置的有限差分导数乘该变量完整范围，再除以同点集基准指标，以便比较 mm 与无量纲变量。此缩放不表示整个范围内线性变化，也不是全局敏感度分析。"))
    packing = _plot_radial_packing(figures_dir, payload)
    if packing:
        figures["print_radial_packing"] = str(packing)
        sections.append(_image_section(packing, "为什么环数与间距不能独立增加", "原始层与打印起点的实际半径对照。槽口、隔墙和内外壁需要同时满足；径向示意并不替代完整三维模块审查。"))
    response = _plot_response(figures_dir, payload)
    if response:
        figures["response_surface"] = str(response)
        sections.append(_image_section(response, "双参数响应面", "色块对应探索采样下的已计算参数网格，不代表网格间所有组合均已求解；与密集复核指标可能有采样差异。"))
    candidates = _candidate_rows(payload)
    pareto = _plot_pareto(figures_dir, payload, candidates)
    if pareto:
        figures["pareto"] = str(pareto)
        caption = "搜索先满足场强等性能门槛，再降低 ppm；圆圈仅标记通过几何与性能复核的非支配候选。磁材质量仅作诊断。" if payload.get("requirements") else "历史结果使用场强与 ppm 双目标；未设置本版本的性能门槛，非支配只针对该次有效候选。"
        sections.append(_image_section(pareto, "候选与基准比较", caption))
    baseline_id = str(baseline.get("candidate_id", "baseline"))
    header_record=best_record(payload)
    header_metrics=record_metrics(header_record) or metrics
    cards = []
    for key in ("mean_norm_T", "pp_norm_ppm", "mass_magnet_kg", "magnet_count"):
        _, unit, _ = METRIC_LABELS[key]
        label = _metric_label(key, payload)
        cards.append("<div class='metric'><span>" + _esc(label) + "</span><strong>" + _fmt(_metric(header_metrics, key)) + "<small> " + _esc(unit) + "</small></strong></div>")
    cube = _number(header_record.get("design", {}).get("cube_mm"))
    rings = header_record.get("design", {}).get("rings", [])
    geometry_text = "磁环记录 %d 条" % len(rings)
    if cube is not None:
        geometry_text += "；立方磁块边长 " + _fmt(cube) + " mm"
    limitations = model_info.get("limitations", [])
    if isinstance(limitations, str):
        limitations = [limitations]
    required_notes = [
        "当前结果属于 Magpylib 均匀磁化快速解析预测。它不等效于 COMSOL 中 mur=1.05 的材料回复响应；模型间差异需要实测对标。",
        "本工作流没有修改原始 .mph 文件。仅创建或导出对标输入不代表完成 COMSOL FEM 求解、网格收敛或一致性验证。",
        "保护装配通道尺寸以本次制造配置为准；各候选裸磁体通孔见布局图。两者因磁槽余量与内壁而不同。旧资料中的220 mm装配内孔尚未核实。",
        (evaluation_label + "是本次唯一成像评价域。平均值与 RMS 按圆盘面积加权；边界采样只用于极值检查。" if planar else evaluation_label + "与" + cylinder_label + "是不同评价域；应按各自的采样点和指标分别比较。"),
        ("峰峰不均匀度为 (圆盘内最大值−圆盘内最小值)/面积平均 |B|×10⁶。各指标分别检查平面圆盘内部与边界的极值，有限采样及局部搜索不证明已找到连续域全局极值。" if planar else "峰峰不均匀度为 (最大值−最小值)/|平均值|×10⁶；该报告基于有限采样点，未证明连续空间极值已收敛。"),
        "磁材质量只包括永磁材料。候选搜索未证明全局最优，仿真结果不构成制造验收或临床成像性能验证。",
    ]
    notes = "".join("<li>" + _esc(note) + "</li>" for note in required_notes + list(limitations))
    info_json = json.dumps(_json_safe(model_info), ensure_ascii=False, indent=2)
    comsol_label = "COMSOL：" + _comsol_status(model_info)
    geometry_json = json.dumps(_json_safe(baseline.get("geometry_check", {})), ensure_ascii=False, indent=2)
    pareto_flags = _pareto_flags(candidates) if candidates else np.zeros(0, dtype=bool)
    summary = {
        "baseline": _json_safe(baseline), "model_info": _json_safe(model_info),
        "manufacturing": _json_safe(payload.get("manufacturing", {})),
        "selected_candidate_id":header_record.get("candidate_id"), "evolution":_json_safe(payload.get("evolution",{})),
        "requirements": _json_safe(payload.get("requirements", {})),
        "performance_summary": _json_safe(payload.get("performance_summary", {})),
        "deliverables": _json_safe(payload.get("deliverables", {})),
        "figures": {k:os.path.relpath(v,str(output_dir)).replace(os.sep,"/") for k,v in figures.items()}, "candidate_count": len(candidates),
        "pareto_ids": [str(row.get("candidate_id")) for i, row in enumerate(candidates) if pareto_flags[i]],
        "report_kind": "快速解析预测；COMSOL状态以model_info为准",
        "evaluation_geometry": "disk" if planar else "legacy_sphere",
        "evaluation_measure": "area" if planar else "volume",
        "optimization_objectives": (["satisfy geometry and performance requirements", "minimize pp_norm_ppm"] if payload.get("requirements") else ["maximize mean_norm_T", "minimize pp_norm_ppm"]),
        "diagnostic_metrics": ["mass_magnet_kg"],
    }
    summary_path = details_dir / "report_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    document = """<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>MRI 磁体快速迭代报告</title><style>
*{box-sizing:border-box}body{margin:0;background:#eef3f8;color:#20324a;font-family:'Microsoft YaHei','Noto Sans CJK SC',Arial,sans-serif;line-height:1.75}main{max-width:1580px;margin:0 auto;padding:32px 22px 72px}header{background:#142b45;border-radius:18px;padding:36px;color:#f8fafc}h1{margin:8px 0 14px;font-size:30px;line-height:1.3}h2{font-size:21px;margin:0 0 18px}h3{font-size:17px}p{margin:10px 0}.eyebrow{font-size:13px;letter-spacing:2px;color:#a8c8e7}.badge{display:inline-block;font-size:12px;margin-right:7px;border-radius:30px;padding:4px 12px;background:#d8e9f7;color:#184b73}.badge.pending{background:#fff0cf;color:#785017}.cards{display:grid;grid-template-columns:repeat(4,1fr);gap:13px;margin-top:24px}.metric{border:1px solid #3b536b;background:#203b58;border-radius:12px;padding:15px}.metric span{font-size:12px;color:#bfd0e1}.metric strong{display:block;font-size:25px;font-weight:600;margin-top:7px}.metric small{font-size:13px;font-weight:400}section{background:#fff;margin-top:22px;padding:28px;border-radius:14px;box-shadow:0 2px 12px #102a4510}section img{display:block;width:100%;height:auto}.caption{color:#61758b;font-size:13px}.notice{border-left:4px solid #d7a143;background:#fff9ed;padding:14px 18px;margin-top:20px;border-radius:6px}.table-wrap{overflow-x:auto}table{width:100%;border-collapse:collapse;font-size:14px}th,td{text-align:left;padding:11px 12px;border-bottom:1px solid #e2e8f0}th{font-weight:600;background:#f4f7fa}td:nth-child(2){font-variant-numeric:tabular-nums}tr.baseline td{background:#fff5d8;font-weight:600}ul{padding-left:22px}li{margin:9px 0}code,pre{font-family:Consolas,monospace;font-size:12px}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f4f7fa;padding:15px;border-radius:8px}details{margin-top:14px}summary{cursor:pointer;color:#285d87;font-weight:600}footer{color:#61758b;text-align:center;margin-top:30px;font-size:12px}@media(max-width:760px){main{padding:18px 12px 40px}header,section{padding:21px}.cards{grid-template-columns:repeat(2,1fr)}h1{font-size:25px}.metric strong{font-size:21px}}@media print{body{background:white}main{padding:0}header{print-color-adjust:exact}section{break-inside:avoid;box-shadow:none;border:1px solid #ddd}details{display:block}}
</style></head><body><main><header><div class="eyebrow">MRI 磁铁排布原理2 / 计算记录</div><h1>永磁主场 · 快速迭代报告</h1><span class="badge">快速解析预测</span><span class="badge pending">""" + _esc(comsol_label) + "</span><p>" + "本轮选中方案：" + _esc(validation_label(header_record)) + " · " + _esc(geometry_text) + "</p><p>" + _esc(evaluation_label) + "</p><div class='cards'>" + "".join(cards) + "</div></header>"
    document += _hero_section(output_dir,payload)
    document += _evolution_section(output_dir,payload)
    document += "<section><h2>候选方案核心对比</h2>" + _candidate_table(candidates, baseline_id, payload) + "</section>"
    document += _performance_section(payload, candidates)
    domain_text = "主评价域：" + evaluation_label + ("。平均场与 RMS 按圆盘面积计算。" if planar else "。辅助评价域：" + cylinder_label + "。")
    document += "<section><h2>本次评价口径与对标状态</h2><p>" + _esc(domain_text) + "主场方向约定沿 z 轴；原始 Bz 保留正负号。</p>"
    document += "<div class='notice'>本页所有数值和图像取自本次实际计算。解析预测与 COMSOL FEM 的完成状态分别记录，不能把模型导出视为 FEM 验证通过。</div>"
    document += "<p><strong>" + _esc(comsol_label) + "</strong></p>" + _cached_line_table(model_info) + "<details><summary>展开模型与对标状态详细记录</summary><pre>" + _esc(info_json) + "</pre></details></section>"
    document += "<details class='audit-details'><summary>展开原始对照与制造审查</summary>"
    document += "<section><h2>原始物理对照指标</h2>" + _metric_table(metrics, payload)
    if baseline.get("validation_metrics") and any(baseline["validation_metrics"].get(key) != metrics.get(key) for key in METRIC_LABELS):
        document += "<h3>独立验证采样指标</h3>" + _metric_table(baseline["validation_metrics"], payload)
    if not planar and model_info.get("baseline_cylinder_metrics"):
        document += "<h3>辅助区域 · " + _esc(cylinder_label) + "</h3>" + _metric_table(model_info["baseline_cylinder_metrics"])
    document += "<details><summary>基准几何检查</summary><pre>" + _esc(geometry_json) + "</pre></details></section>"
    document += _manufacturing_section(payload, candidates)+"</details>"
    document += "<details><summary>全部候选布局、CLASS与诊断方案</summary>"+_deliverables_section(output_dir, payload)+"</details>"
    document += "<details><summary>探索曲线、敏感度与其他图表</summary>"+"".join(sections)+"</details>"
    document += "<section><h2>如何理解这些结果</h2><ul>" + notes + "</ul></section>"
    document += "<footer>PNG 已同时独立保存。本 HTML 内嵌全部图像，可离线打开；不加载远程字体、脚本或图像。</footer></main></body></html>"
    html_path = output_dir / "report_中文.html"
    html_path.write_text(document, encoding="utf-8")
    return {"report_html": str(html_path), "summary_json": str(summary_path), "figures": figures}
