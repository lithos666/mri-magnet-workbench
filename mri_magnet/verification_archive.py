"""Append later FEM evidence without rerunning or overwriting optimization."""
from pathlib import Path
import csv,html,json,re
from .identity import canonical_physical_sha256
from .serialization import save_json


def append_fem_evidence(run_dir,report,report_file):
    root=Path(run_dir).resolve();path=Path(report_file).resolve();path.relative_to(root)
    relative=path.relative_to(root).as_posix();summary=path.with_name('COMSOL验证结论.md')
    comparison=report.get('comparison',{});metrics=comparison.get('fem_metrics') or {}
    fields=[('# COMSOL 验证结论'),'',
        '设计指纹：`%s`。'%report.get('design_sha256','未记录'),
        '状态：%s；FEM收敛：%s；Python模型精度：%s。'%(report.get('status'),
            '通过' if report.get('fem_verified') else '未通过或待完成',
            '通过' if report.get('model_accuracy_passed') else '未通过或待完成'),
        '', '仅对应此设计、材料、评价域、采样及物理模型；CLASS编译成功不代替数值验证。','']
    if metrics:fields.append('FEM平均场：%.6f mT；采样峰峰值：%.6f ppm。'%(metrics['mean_norm_T']*1000,metrics['pp_norm_ppm']))
    if comparison:
        fields.append('Python平均场相对误差：%.6f%%；PPM差：%.6f；允许PPM差：%.6f。'%(
            comparison['mean_relative_error']*100,comparison['ppm_absolute_error'],comparison['ppm_error_limit']))
    fields+=['','[完整机器可读报告](fem_report.json)']
    summary.write_text('\n'.join(fields)+'\n',encoding='utf-8')
    config_path=root/'配置快照.json'
    config=json.loads(config_path.read_text(encoding='utf-8-sig')) if config_path.exists() else {
        'physics':report.get('physics',{}),'target':report.get('domain',{}),'polarization_model':'remanence'}
    from .workbench_archive import physics_status
    for design_file in (root/'候选方案').glob('*/设计.json'):
        design=json.loads(design_file.read_text(encoding='utf-8-sig'))
        if canonical_physical_sha256(design)!=report.get('design_sha256'):continue
        checked=design_file.with_name('复核指标.json')
        if checked.exists():
            data=json.loads(checked.read_text(encoding='utf-8-sig'))
            status=physics_status({'design':design,'physics_validation':report},config)
            data.update(fem_verified=status['fem_verified'],physics_status=status,
                model_accuracy_passed=status['model_accuracy_passed'],
                fem=metrics or None,physics_validation=report)
            save_json(checked,data)
    manifest_path=root/'归档清单.json';matched_layouts=set();best_status=None
    if manifest_path.exists():
        manifest=json.loads(manifest_path.read_text(encoding='utf-8-sig'))
        rows=list(manifest.get('candidates',[]))+[manifest.get('best')]
        matched_ids=set()
        for row in rows:
            if not isinstance(row,dict):continue
            if row.get('design') and canonical_physical_sha256(row['design'])==report.get('design_sha256'):
                row['physics_validation']=report;row['physics_status']=physics_status(row,config)
                row['fem_verified']=row['physics_status']['fem_verified']
                row['model_accuracy_passed']=row['physics_status']['model_accuracy_passed']
                if row.get('layout_file'):matched_layouts.add(row['layout_file'])
                if row.get('candidate_id'):matched_ids.add(row['candidate_id'])
                if row is manifest.get('best'):best_status=row['physics_status']
        for row in manifest.get('pareto_recommendations',[]):
            if row.get('candidate_id') in matched_ids:
                status=physics_status({'design':report['design'],'physics_validation':report},config) if report.get('design') else None
                if status:row.update(physics_validation=report,physics_status=status)
        if best_status:
            manifest.update(fem_verified=best_status['fem_verified'],model_accuracy_passed=best_status['model_accuracy_passed'])
        manifest.setdefault('artifacts',{}).update(fem_report=relative,fem_summary=summary.relative_to(root).as_posix())
        save_json(manifest_path,manifest)
    comparison_path=root/'方案对比.csv'
    if comparison_path.exists() and matched_layouts:
        with comparison_path.open(encoding='utf-8-sig',newline='') as stream:
            reader=csv.DictReader(stream);columns=reader.fieldnames;rows=list(reader)
        status=physics_status({'design':report.get('design'),'physics_validation':report},config)
        for row in rows:
            if row.get('布局图') in matched_layouts:
                if 'FEM复核' in row:row['FEM复核']='已验证' if status['fem_verified'] else '未通过或待验证'
                if 'Python模型精度' in row:row['Python模型精度']='通过' if status['model_accuracy_passed'] else '未通过或待标定'
        with comparison_path.open('w',encoding='utf-8-sig',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=columns);writer.writeheader();writer.writerows(rows)
    note=root/'本次运行速览.md';link=summary.relative_to(root).as_posix()
    if note.exists():
        marker='<!-- v6-fem:%s -->'%report.get('job_id',relative)
        text=note.read_text(encoding='utf-8')
        if best_status:
            accuracy='通过' if best_status['model_accuracy_passed'] else '未通过'
            fem='已收敛验证' if best_status['fem_verified'] else '未通过或待验证'
            text=re.sub(r'性能门槛：([^；。\n]*)；Python模型精度：[^；\n]*；FEM：[^。\n]*。',
                lambda match:'性能门槛：'+match[1]+'；Python模型精度：'+accuracy+'；FEM：'+fem+'。',text)
            if metrics and best_status['fem_verified']:
                text=re.sub(r'^\| COMSOL FEM \|.*$',lambda match:'| COMSOL FEM | %.6f | %.6f | 同一冻结%d点集 |'%(
                    metrics['mean_norm_T']*1000,metrics['pp_norm_ppm'],metrics.get('n_points',0)),text,flags=re.M)
            note.write_text(text,encoding='utf-8')
        if marker not in text:
            text+='\n\n'+marker+'\n## 后续 COMSOL 验证记录\n\n'
            text+='设计 `%s`：%s。[查看结论](%s)。\n'%(report.get('design_sha256','')[:12],report.get('status'),link)
            note.write_text(text,encoding='utf-8')
    page=root/'报告.html'
    if page.exists():
        text=page.read_text(encoding='utf-8')
        if matched_layouts:
            def update_card(match):
                card=match[0]
                if any('href="'+html.escape(layout,quote=True)+'"' in card for layout in matched_layouts):
                    status=physics_status({'design':report.get('design'),'physics_validation':report},config)
                    label='；模型精度'+('通过' if status['model_accuracy_passed'] else '未通过')+'；FEM '+('已收敛' if status['fem_verified'] else '待复核')
                    card=card.replace('；模型精度待标定；FEM 待复核',label)
                return card
            text=re.sub(r'<article>.*?</article>',update_card,text,flags=re.S)
            page.write_text(text,encoding='utf-8')
        marker='<!-- v6-fem:%s -->'%report.get('job_id',relative)
        if marker not in text:
            block=marker+'<p><strong>后续 COMSOL 验证</strong>：'+html.escape(str(report.get('status')))
            block+='；设计 '+html.escape(str(report.get('design_sha256',''))[:12])+'。<a href="'+html.escape(link)+'">查看验证结论</a></p>'
            text=text.replace('</header>',block+'</header>',1)
            text=text.replace('解析计算结果；FEM 待验证；全局最优未证明。','数值模型结果；FEM最新结论见追加记录；全局最优未证明。')
            page.write_text(text,encoding='utf-8')
    return {'fem_report':relative,'fem_summary':link}
