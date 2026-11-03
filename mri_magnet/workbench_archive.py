"""Readable workbench deliverables from already evaluated records; no field solves."""
from __future__ import annotations

import csv
import html
import json
import math
from pathlib import Path

from .de_novo import limits_from_config
from .geometry import Design
from .layout import render_rz_layout
from .manufacturing import audit_manufacturing
from .objectives import (balanced_components, check_performance, feasibility_score,
                         pareto_front, pareto_roles, scoring_options)
from .parameterization import topology_identity
from .serialization import jsonable, save_json


_POLARITY = {"out": "N径向朝外", "in": "N径向朝内", "down": "N轴向-Z", "up": "N轴向+Z"}


def physics_status(row, config):
    """Independent accuracy/FEM flags; exporting a class never verifies either."""
    from .identity import canonical_physical_sha256
    from .field_models import model_fingerprint
    physics = config.get('physics') or {'field_model':'legacy'}
    coupled = physics.get('field_model') == 'coupled'
    verification = row.get('physics_validation') or {}
    binding_errors=[]
    try:
        design_matches=bool(row.get('design')) and verification.get('design_sha256')==canonical_physical_sha256(row['design'])
    except (ValueError,TypeError,KeyError,AttributeError):
        design_matches=False
    if not design_matches:binding_errors.append('verification design binding is missing or mismatched')
    try:
        model_matches=verification.get('model_fingerprint')==model_fingerprint(config)
    except (ValueError,TypeError,KeyError,AttributeError):
        model_matches=False
    if not model_matches:binding_errors.append('verification model binding is missing or mismatched')
    if not isinstance(verification.get('domain'),dict) or verification['domain']!=config.get('target'):
        binding_errors.append('verification domain binding is missing or mismatched')
    binding_valid=not binding_errors
    return {'physics':dict(physics), 'field_model':'coupled' if coupled else 'legacy',
            'model_label':('阵列自洽互耦 · %d阶积分 · 每轴%d单元' %
                (physics.get('quadrature_order',2),physics.get('cells_per_axis',1))) if coupled
                else '固定源 · '+config.get('polarization_model','remanence'),
            'model_accuracy_passed':binding_valid and verification.get('model_accuracy_passed') is True,
            'fem_verified':binding_valid and verification.get('fem_verified') is True,
            'verification_binding_valid':binding_valid,'verification_binding_errors':binding_errors,
            'verification':verification}


def metric_comparison_rows(row, config):
    """Same-design models and point sets remain distinguishable in reports."""
    status=physics_status(row,config);verification=status['verification']
    return [('旧快速模型对照',row.get('legacy_metrics') or verification.get('legacy_metrics')),
            (('耦合' if status['field_model']=='coupled' else '固定源')+'训练',row.get('training_metrics')),
            (('耦合' if status['field_model']=='coupled' else '固定源')+'密集复核',_metrics(row)),
            ('COMSOL FEM',verification.get('fem_metrics') if status['fem_verified'] else None)]


def _fmt(value, digits=4):
    if value is None:
        return "—"
    number = float(value)
    if not math.isfinite(number):
        return "—"
    return ("%.*f" % (digits, number)).rstrip("0").rstrip(".") or "0"


def _metrics(row):
    return row.get("validation_metrics") or row.get("dense_metrics") or row.get("metrics") or {}


def _domain(config):
    target = config["target"]
    if target["geometry"] == "sphere":
        return "球形 DSV，直径 %s mm，均值按体积积分" % _fmt(target["diameter_mm"])
    return "z=%s mm 平面圆形 FOV，直径 %s mm，均值按面积积分" % (
        _fmt(target.get("z_mm", 0.)), _fmt(target["diameter_mm"]))


def _requirements(config):
    target = config["target"]
    maximum = target.get('max_ppm')
    return {"min_mean_norm_T": float(target["min_field_mT"])*1e-3,
            "max_pp_norm_ppm": None if maximum is None else float(maximum), "field_tolerance_T": 0.,
            "field_scale_T": max(float(target["min_field_mT"])*1e-3, 1e-12),
            "ppm_scale": max(float(maximum), 1.) if maximum is not None else 1000.,
            "scoring": scoring_options(config.get('scoring'))}


def _reference_for(row, seeds):
    current = Design.from_dict(row["design"])
    signature = topology_identity(current)
    index = row.get("topology_index")
    candidates = []
    if isinstance(index, int) and 1 <= index <= len(seeds):
        candidates.append(seeds[index-1])
    candidates.extend(seed for seed in seeds if seed not in candidates)
    for seed in candidates:
        if topology_identity(Design.from_dict(seed["design"])) == signature:
            return seed
    # A changed topology may still be compared to its declared original seed.
    # Source slot identities make additions/removals explicit below.
    if isinstance(index, int) and 1 <= index <= len(seeds):
        return seeds[index-1]
    return None


def physical_change_rows(design, reference):
    """One row per source slot, including additions and deletions, in millimetres."""
    actual = {(r.layer_index, r.slot_index): r for r in design.rings}
    origin = {(r.layer_index, r.slot_index): r for r in reference.rings} if reference else {}
    rows = []
    for key in sorted(set(actual) | set(origin)):
        now, old = actual.get(key), origin.get(key)
        ring = now or old
        rows.append({"层": "R%02d" % (key[0]+1), "径向槽": key[1]+1,
                     "状态": "保留" if now and old else "新增" if now else "删除",
                     "中心R_mm": now.radius_mm if now else None,
                     "底边R_mm": now.radius_mm-design.cube_mm/2 if now else None,
                     "起点中心R_mm": old.radius_mm if old else None,
                     "起点底边R_mm": old.radius_mm-reference.cube_mm/2 if old else None,
                     "ΔR_mm": now.radius_mm-old.radius_mm if now and old else None,
                     "中心Z_mm": now.z_mm if now else None,
                     "起点中心Z_mm": old.z_mm if old else None,
                     "ΔZ_mm": now.z_mm-old.z_mm if now and old else None,
                     "极性": _POLARITY[ring.polarity], "列数": ring.count})
    return rows


def _write_csv(path, rows, columns=None):
    if columns is None:
        columns = list(rows[0]) if rows else []
    with Path(path).open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader(); writer.writerows(rows)


def _changes_markdown(rows):
    lines = ["| 层 / 槽 | 状态 | 内侧底面 R_底 (mm) | 中心 R_中心 (mm)（ΔR） | 中心 Z mm（ΔZ） | 极性 | 列数 |",
             "|---|---|---:|---:|---:|---|---:|"]
    for row in rows:
        dr = ("%+.3f" % row["ΔR_mm"]) if row["ΔR_mm"] is not None else "—"
        dz = ("%+.3f" % row["ΔZ_mm"]) if row["ΔZ_mm"] is not None else "—"
        lines.append("| %s / %s | %s | %s | %s（%s） | %s（%s） | %s | %s |" % (
            row["层"], row["径向槽"], row["状态"], _fmt(row['底边R_mm'],3), _fmt(row["中心R_mm"], 3), dr,
            _fmt(row["中心Z_mm"], 3), dz, row["极性"], row["列数"]))
    return lines


def _relative_export(export, root):
    result = dict(export)
    for key in ("java_file", "class_file", "manifest_file", "compile_log", "build_log", "build_progress_log"):
        if result.get(key):
            try:
                result[key] = Path(result[key]).resolve().relative_to(root).as_posix()
            except ValueError:
                pass
    result["file_paths_relative_to_run_root"] = True
    return result


def _render_pareto(records, front, roles, labels, requirements, path):
    """Plot evaluated dense metrics only; does not invoke the field engine."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    for row in records:
        metrics = row['metrics']
        ax.scatter(metrics['mean_norm_T']*1e3, metrics['pp_norm_ppm'],
                   color='#A9B4C2', s=28, alpha=.65)
    ax.plot([row['metrics']['mean_norm_T']*1e3 for row in front],
            [row['metrics']['pp_norm_ppm'] for row in front],
            'o-', color='#367BC1', linewidth=1.3, markersize=4, label='已复核帕累托前沿')
    used = set()
    for role, color, marker in (('balanced','#E49628','*'),('high_field','#A95156','^'),
                                ('lowest_ppm','#21968D','v')):
        row = roles.get(role)
        if row is None:
            continue
        metrics = row['metrics']
        ax.scatter(metrics['mean_norm_T']*1e3,metrics['pp_norm_ppm'],s=120,
                   marker=marker,color=color,label=labels[role],zorder=5)
        if id(row) not in used:
            combined = ' / '.join(labels[key] for key, value in roles.items() if value is row)
            ax.annotate(combined,(metrics['mean_norm_T']*1e3,metrics['pp_norm_ppm']),
                        xytext=(8,8),textcoords='offset points',fontsize=9)
            used.add(id(row))
    ax.axvline(requirements['min_mean_norm_T']*1e3,color='#73A78E',linestyle='--',linewidth=1)
    if requirements.get('max_pp_norm_ppm') is not None:
        ax.axhline(requirements['max_pp_norm_ppm'],color='#73A78E',linestyle='--',linewidth=1)
    ax.set(xlabel='体积 / 面积平均场强 [mT]',ylabel='峰峰值不均匀度 [ppm]',
           title='密集复核候选：场强与均匀度的真实权衡')
    ax.grid(alpha=.15);ax.legend(fontsize=9)
    fig.savefig(path,dpi=150);plt.close(fig)


def archive_run(run_dir, config, seeds, validated_records, optimization_result, evolution,
                export_comsol=True, compile_class=None):
    """Write layouts, human-readable metrics, physical deltas and COMSOL models.

    All candidates provided by dense validation remain reviewable, including
    those missing target thresholds. Such candidates are labelled diagnostic,
    and geometrically invalid candidates are never exported as COMSOL models.
    Returned artifact links are relative to ``run_dir`` so it may be renamed.
    """
    root = Path(run_dir).resolve(); root.mkdir(parents=True, exist_ok=True)
    details = root/"详细数据"; details.mkdir(exist_ok=True)
    requirements = _requirements(config)
    limits = limits_from_config(config)
    records = []
    for source in validated_records:
        row = dict(source)
        if not row.get("design") or not _metrics(row):
            continue
        design = Design.from_dict(row["design"])
        row["metrics"] = _metrics(row)
        row["performance_check"] = check_performance(row["metrics"], requirements)
        row["manufacturing_check"] = audit_manufacturing(design, config["manufacturing"], limits)
        row["feasible"] = row["performance_check"]["valid"] and row["manufacturing_check"]["valid"]
        row['physics_status']=physics_status(row,config)
        records.append(row)
    records.sort(key=lambda r: (not r["manufacturing_check"]["valid"],
                               feasibility_score(r["metrics"], requirements),
                               r["metrics"]["pp_norm_ppm"]))
    roles = pareto_roles(records, requirements)
    front = pareto_front(records, requirements)
    role_labels = {'balanced': '均衡推荐' if requirements['scoring']['mode']=='balanced' else '最优均匀',
                   'high_field': '高场强', 'lowest_ppm': '最低PPM'}
    for row in records:
        row['pareto_roles'] = [role for role, selected in roles.items() if selected is row]
        row['is_pareto'] = any(row is item for item in front)
        row['ranking_score'] = feasibility_score(row['metrics'], requirements)
        if row['feasible'] and requirements['scoring']['mode']=='balanced':
            row['balanced_components'] = balanced_components(row['metrics'], requirements)
    artifacts = {"overview": "本次运行速览.md", "report": "report_中文.html",
                 "comparison": "方案对比.csv", "manifest": "归档清单.json"}
    delivered, comparison = [], []
    for index, row in enumerate(records):
        design = Design.from_dict(row["design"])
        metrics = row["metrics"]
        if row['feasible']:
            label = next((role_labels[role] for role in ('balanced','lowest_ppm','high_field')
                          if role in row['pareto_roles']), '帕累托备选' if row['is_pareto'] else '备选')
        else:
            label = '诊断未达标'
        name = "%02d_%s_%smT_%sppm_%d块" % (index+1, label,
            _fmt(metrics["mean_norm_T"]*1e3, 2), _fmt(metrics["pp_norm_ppm"], 3), design.magnet_count)
        folder = root/"候选方案"/name; folder.mkdir(parents=True, exist_ok=True)
        prefix = folder.relative_to(root).as_posix()
        reference = _reference_for(row, seeds)
        reference_design = Design.from_dict(reference["design"]) if reference else None
        layout = folder/"磁铁布局_RZ.png"
        domain_label=('球形 DSV' if config['target']['geometry']=='sphere' else '平面 FOV')+' Φ%g mm'%config['target']['diameter_mm']
        render_rz_layout(design, layout, title="MRI · %s · %s · %s mT / %s ppm" % (
            label, domain_label, _fmt(metrics["mean_norm_T"]*1e3, 6), _fmt(metrics["pp_norm_ppm"], 6)),
            manufacturing_audit=row["manufacturing_check"], reference_design=reference_design,
            reference_label='合成起点' if config.get('mode')=='de_novo' else '打印投影起点')
        save_json(folder/"设计.json", design.to_dict())
        status=row['physics_status']
        save_json(folder/"复核指标.json", {"training": row.get("training_metrics"), "dense": metrics,
                  "performance": row["performance_check"], "manufacturing": row["manufacturing_check"],
                  "domain": config["target"], "physics_status":status,
                  "legacy":row.get('legacy_metrics') or status['verification'].get('legacy_metrics'),
                  "fem":status['verification'].get('fem_metrics'),
                  "fem_verified":status['fem_verified'],"model_accuracy_passed":status['model_accuracy_passed']})
        changes = physical_change_rows(design, reference_design)
        save_json(folder/'尺寸改动_毫米.json',{'units':'mm','radial_label':'inner face radius = center radius - cube_mm/2',
            'cube_mm':design.cube_mm,'rows':changes})
        _write_csv(folder/"尺寸改动_毫米.csv", changes)
        (folder/"尺寸改动_毫米.md").write_text("\n".join(["# 物理尺寸改动", "",
            "R_底为径向内侧底面半径，与布局图色块数字口径一致；R_中心=R_底+磁块边长/2。Z为磁块中心坐标，单位mm。图纸按可读精度显示，精确坐标见设计JSON。", ""] +
            _changes_markdown(changes)), encoding="utf-8")
        model = None
        if export_comsol and row["manufacturing_check"]["valid"]:
            from .comsol_export import export_comsol_model
            settings = dict(config.get("deliverables", {}))
            settings.pop("enabled", None)
            if compile_class is not None:
                settings["compile"] = bool(compile_class)
            settings["runtime_validate"] = False
            model = export_comsol_model(design, folder, config.get("comsol_root", ""), settings)
            manifest_path = Path(model["manifest_file"])
            model = _relative_export(model, root)
            save_json(manifest_path, model)
        row.update({"folder": prefix, "layout_file": prefix+"/磁铁布局_RZ.png",
                    "changes_file": prefix+"/尺寸改动_毫米.csv", "model_export": model,
                    "archive_label": label, "reference_seed_id": reference.get("id") if reference else None})
        delivered.append(row)
        comparison.append({"方案": label, "平均场_mT": metrics["mean_norm_T"]*1e3,
            "峰峰值_ppm": metrics["pp_norm_ppm"], "磁环数": len(design.rings), "磁铁块数": design.magnet_count,
            "帕累托前沿": "是" if row['is_pareto'] else "否",
            "候选角色": "、".join(role_labels[role] for role in row['pareto_roles']),
            "制造审查": "通过" if row["manufacturing_check"]["valid"] else "冲突",
            "性能门槛": "通过" if row["performance_check"]["valid"] else "未通过",
            "布局图": row["layout_file"], "尺寸改动表": row["changes_file"],
            "Java": model.get("java_file") if model else "",
            "CLASS": model.get("class_file") if model else "",
            "导出状态": model["status"] if model else "未导出",
            "求场模型":status['model_label'],
            "Python模型精度":'通过' if status['model_accuracy_passed'] else '未通过或待标定',
            "FEM复核":'已验证' if status['fem_verified'] else '待验证'})
    _write_csv(root/artifacts["comparison"], comparison,
               list(comparison[0]) if comparison else ["方案", "平均场_mT", "峰峰值_ppm"])
    best = delivered[0] if delivered else None
    if best:
        artifacts["layout"] = best["layout_file"]
        artifacts["physical_changes"] = best["changes_file"]
        model = best.get("model_export") or {}
        if model.get("java_file"):
            artifacts["java"] = model["java_file"]
        if model.get("class_file"):
            artifacts["class"] = model["class_file"]
    baseline = _reference_for(best, seeds) if best else (seeds[0] if seeds else None)
    gate_label = '性能双门槛' if config['target'].get('max_ppm') is not None else '场强硬门槛'
    conclusion = (("本轮均衡推荐方案通过制造审查与%s。" if requirements['scoring']['mode']=='balanced' else
                   "本轮最优方案通过制造审查与%s。") % gate_label if best and best["feasible"] else
                  "本轮未取得双门槛通过方案，以下保留诊断排布供比较。" if best else
                  "本轮没有可归档的密集复核候选，请查看前置校验和合成诊断。")
    if config['target'].get('max_ppm') is None and best and not best['feasible']:
        conclusion = '本轮未取得场强硬门槛通过方案，以下保留诊断排布供比较。'
    target_description = '目标：平均场 ≥ %s mT，' % _fmt(config['target']['min_field_mT'])
    target_description += ('峰峰值 ≤ %s ppm。' % _fmt(config['target']['max_ppm'])
                           if config['target'].get('max_ppm') is not None else
                           '峰峰值不设硬上限，作为连续优化目标。')
    lines = ["# 本次运行速览", "", conclusion, "", "评价域："+_domain(config)+"。", "",
             target_description, "",
             "## 起点与最优解", "", "| 排布 | 平均场 mT | 峰峰值 ppm | 磁铁块数 | 点集 | 状态 |",
             "|---|---:|---:|---:|---|---|"]
    for name, row in (("对应拓扑起点", baseline), ("本轮最优", best)):
        if row:
            m = _metrics(row)
            check = check_performance(m, requirements)
            lines.append("| %s | %s | %s | %s | %s | %s |" % (
                name, _fmt(m["mean_norm_T"]*1e3, 6), _fmt(m["pp_norm_ppm"], 6),
                Design.from_dict(row["design"]).magnet_count, m.get("pointset", "未记录"),
                gate_label+'通过' if check["valid"] else "未达标"))
    if baseline and best:
        a, b = _metrics(baseline), best["metrics"]
        if a.get("pointset") and a.get("pointset") == b.get("pointset"):
            delta = (b["mean_norm_T"]-a["mean_norm_T"])*1e3
            decrease = (1-b["pp_norm_ppm"]/a["pp_norm_ppm"])*100 if a["pp_norm_ppm"] > 0 else None
            lines.extend(["", "相对同点集起点：ΔB=%+.6f mT，ppm 下降 %s%%。" % (delta, _fmt(decrease, 3))])
        else:
            lines.extend(["", "起点与最佳若使用不同点集，表中明确列出；不据此计算改进百分比。"])
    if best:
        status=best['physics_status']
        lines.extend(['','## 物理模型与独立验收','',
            '本轮求场：'+status['model_label']+'。',
            '性能门槛：'+('通过' if best['performance_check']['valid'] else '未通过')+
            '；Python模型精度：'+('通过' if status['model_accuracy_passed'] else '未通过或待标定')+
            '；FEM：'+('已验证' if status['fem_verified'] else '待验证')+'。','',
            '| 同一排布的计算口径 | 平均场 mT | 峰峰值 ppm | 点集 |',
            '|---|---:|---:|---|'])
        for label,m in metric_comparison_rows(best,config):
            lines.append('| %s | %s | %s | %s |'%(label,
                _fmt(m['mean_norm_T']*1e3,6) if m else '—',
                _fmt(m['pp_norm_ppm'],6) if m else '—',m.get('pointset','未记录') if m else '未计算或未通过验证'))
        lines.extend(['','模型精度门槛：相对收敛COMSOL平均场偏差≤0.5%；PPM差≤max(10 ppm，COMSOL值的10%)。',
            '不同点集或不同物理模型的指标不直接混合排序；未计算的对照不以其他口径代填。'])
    recommendations=[]
    for role in ('balanced','high_field','lowest_ppm'):
        selected = roles.get(role)
        if selected is None:
            continue
        model = selected.get('model_export') or {}
        recommendations.append({'role':role,'label':role_labels[role],
            'candidate_id':selected.get('candidate_id'),'metrics':selected['metrics'],
            'layout_file':selected['layout_file'],'folder':selected['folder'],
            'class_file':model.get('class_file'),'java_file':model.get('java_file')})
    if recommendations:
        lines.extend(['','## 帕累托候选对比','',
            '| 角色 | 平均场 mT | 峰峰值 ppm | 磁铁块数 | 布局图 |',
            '|---|---:|---:|---:|---|'])
        for rec in recommendations:
            selected=roles[rec['role']];m=rec['metrics']
            lines.append('| %s | %s | %s | %s | [打开](%s) |' % (
                rec['label'],_fmt(m['mean_norm_T']*1e3,6),_fmt(m['pp_norm_ppm'],6),
                Design.from_dict(selected['design']).magnet_count,rec['layout_file']))
        lines.extend(['','角色可由同一排布同时承担；表中只比较本轮完成密集复核且通过硬门槛的真实候选。'])
    if requirements['scoring']['mode']=='balanced':
        score = requirements['scoring']
        lines.extend(['','均衡评价：先满足场强及已设置的均匀度硬门槛，再最小化',
            '`(1-w)·ln(1+ppm/ppm_scale) - w·ln(1+(B/Bmin-1)/gain_scale)`。',
            '本轮 w=%s，场强增益尺度=%s%%，ppm尺度=%s ppm；两项均采用递减收益。' % (
                _fmt(score['field_weight'],3),_fmt(score['field_gain_scale']*100,3),_fmt(score['ppm_scale'],3)),
            '该推荐只表达上述公开权衡；最低PPM和最高场强候选分别保留，不等于全局最优证明。'])
    lines.extend(["", "## 算法阶段贡献", "", "| 阶段 | 平均场 mT | ppm | 求场次数 | 口径 |",
                  "|---|---:|---:|---:|---|"])
    summaries = (optimization_result or {}).get("metadata", {}).get("stage_summaries", [])
    topology = best.get("topology_index") if best else None
    # Chronological per-round milestones avoid attributing a later restart's
    # improvement to an earlier DE stage, or hiding a coordinate recovery step.
    stages=[]
    if baseline:
        stages.append(('拓扑合成',dict(baseline,metrics=baseline.get('training_metrics',baseline.get('metrics',{})))))
    labels={'initialization':'混合初始化','de':'DE 演化','local':'坐标精修',
            'paired':'成对补偿精修','local_recovery':'补偿余量回收精修','active_set':'灵敏度 LP 精修'}
    for summary in sorted(summaries,key=lambda s:s.get('global_evaluations',0)):
        if topology is not None and summary.get('topology_index')!=topology:continue
        phase=summary.get('stage',summary.get('phase'))
        if phase not in labels or not summary.get('metrics'):continue
        label=('第%d轮 · '%summary['island_round'] if summary.get('island_round') else '')+labels[phase]
        stages.append((label,summary))
    for label,candidate in stages:
        if candidate:
            m = candidate["metrics"]
            lines.append("| %s | %s | %s | %s | %s |" % (label,
                _fmt(m["mean_norm_T"]*1e3, 6), _fmt(m["pp_norm_ppm"], 6),
                candidate.get("global_evaluations", "—"),m.get("pointset", "训练点集未记录")))
        else:
            lines.append("| %s | — | — | — | 未执行或未记录 |" % label)
    recorded_phases={s.get('stage',s.get('phase')) for s in summaries
                     if topology is None or s.get('topology_index')==topology}
    for phase in ('initialization','de','local','paired','active_set'):
        if phase not in recorded_phases:
            lines.append('| %s | — | — | — | 未执行或未记录 |'%labels[phase])
    if best:
        lines.append("| 密集复核 | %s | %s | 独立统计 | 密集点集与连续极值细化 |" % (
            _fmt(best["metrics"]["mean_norm_T"]*1e3, 6), _fmt(best["metrics"]["pp_norm_ppm"], 6)))
        lines.extend(["", "[打开最优布局图](%s) · [逐磁环毫米尺寸表](%s)" % (
            best["layout_file"], best["changes_file"]), "", "## 各磁环物理改动", ""])
        lines.extend(_changes_markdown(physical_change_rows(Design.from_dict(best["design"]),
            Design.from_dict(baseline["design"]) if baseline else None)))
    lines.extend(["", "磁场指标采用本轮配置的求场模型；COMSOL 文件导出或编译不代表 FEM 网格、求解或精度复核已经完成。",
                  "全局最优未证明。训练、密集复核、监控预览与基函数预计算的求场成本分开记录。", ""])
    (root/artifacts["overview"]).write_text("\n".join(lines), encoding="utf-8")
    esc = lambda value: html.escape(str(value), quote=True)
    cards = []
    for row in delivered:
        m = row["metrics"]
        cards.append('<article><h2>%s · %s mT / %s ppm</h2><p>%s</p><a href="%s"><img src="%s" alt="R–Z磁铁布局图"></a>'
                     '<p><a href="%s">毫米尺寸表</a> · <a href="%s/设计.json">精确设计</a></p></article>' % (
            esc(row["archive_label"]), esc(_fmt(m["mean_norm_T"]*1e3, 6)), esc(_fmt(m["pp_norm_ppm"], 6)),
            esc(row['physics_status']['model_label']+'；'+(gate_label+'通过' if row['feasible'] else '诊断方案：未满足全部门槛')+
                '；模型精度'+('通过' if row['physics_status']['model_accuracy_passed'] else '待标定')+
                '；FEM '+('已验证' if row['physics_status']['fem_verified'] else '待复核')),
            esc(row["layout_file"]), esc(row["layout_file"]), esc(row["changes_file"]), esc(row["folder"])))
    media = ""
    evolution = evolution or {}
    gif = evolution.get("gif_file")
    timeline = evolution.get("timeline_file")
    if gif:
        media += '<h2>演化回放</h2><img src="%s" alt="迭代演化快照">' % esc(gif)
        artifacts["evolution_gif"] = gif
    if timeline:
        media += '<p><a href="%s">打开时间轴滑块回放</a></p>' % esc(timeline)
        artifacts["evolution_timeline"] = timeline
    if evolution.get('convergence_file'):
        artifacts['convergence']=evolution['convergence_file']
        media+='<h2>求解器收敛记录</h2><img src="%s" alt="训练点收敛曲线">'%esc(evolution['convergence_file'])
    if front:
        artifacts['pareto'] = '帕累托前沿.png'
        _render_pareto(records,front,roles,role_labels,requirements,root/artifacts['pareto'])
        media = '<h2>已复核帕累托候选</h2><img src="帕累托前沿.png" alt="场强与均匀度帕累托前沿">'+media
    diagnostics = {"config": config, "seeds": seeds, "optimization_metadata": (optimization_result or {}).get("metadata", {}),
                   "evolution": evolution, "dense_candidates": delivered}
    page = '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">' \
           '<title>MRI 本次运行报告</title><style>body{font:15px Microsoft YaHei,sans-serif;background:#f2f5fa;color:#24344b;margin:0}' \
           'main{max-width:1500px;margin:auto;padding:28px}article,header,details{background:white;padding:22px;margin:16px 0;border-radius:12px}' \
           'img{max-width:100%%;height:auto}a{color:#1b65a8}pre{white-space:pre-wrap;overflow-wrap:anywhere}h1{margin-top:0}</style><main>' \
           '<header><h1>MRI 磁体优化结果</h1><p>%s</p><p>%s</p><p><a href="本次运行速览.md">本次运行速览</a> · ' \
           '<a href="方案对比.csv">全部方案对比</a></p><p>解析计算结果；FEM 待验证；全局最优未证明。</p></header>%s%s' \
           '<details><summary>配置、算法阶段与审查原始记录</summary><pre>%s</pre></details></main></html>' % (
               esc(conclusion), esc(_domain(config)), "".join(cards), media,
               html.escape(json.dumps(jsonable(diagnostics), ensure_ascii=False, indent=2, allow_nan=False)))
    (root/artifacts["report"]).write_text(page, encoding="utf-8")
    result = {"status": "verified_target_met" if best and best["feasible"] else "diagnostic_only" if best else "no_candidates",
              "artifacts": artifacts, "best": best, "candidates": delivered,
              "pareto_recommendations": recommendations,
              "pareto_front_ids": [r.get('candidate_id') for r in front],
              "scoring": requirements['scoring'],
              "fem_verified": bool(best and best['physics_status']['fem_verified']),
              "model_accuracy_passed":bool(best and best['physics_status']['model_accuracy_passed']),
              "physics":dict(config.get('physics') or {'field_model':'legacy'}),"global_optimum_proven": False,
              "paths_relative_to_run_root": True, "domain": config["target"]}
    save_json(root/artifacts["manifest"], result)
    save_json(details/"archive_inputs.json", diagnostics)
    return result
