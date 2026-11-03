"""Compare fixed precision profiles against one converged FEM reference."""
from copy import deepcopy
from pathlib import Path
import json,time,hashlib
from .serialization import save_json
from .metrics import field_metrics
from .identity import canonical_physical_sha256


def calibrate_design(design,config,sample,output_dir,fem_report=None,event_callback=None,cancel_check=None):
    from .field_models import make_field_engine,model_fingerprint
    from .comsol_validation import prepare_fem_validation,run_fem_validation,compare_fem_fields,sample_fingerprint
    from .comsol_bridge import read_field_csv
    folder=Path(output_dir);folder.mkdir(parents=True,exist_ok=True)
    if isinstance(fem_report,(str,Path)):
        fem_report=json.loads(Path(fem_report).read_text(encoding='utf-8-sig'))
    if fem_report is None:
        job=prepare_fem_validation(design,sample,folder/'FEM',config['comsol_root'],settings=config.get('fem_validation'))
        fem_report=run_fem_validation(job,event_callback=event_callback,cancel_check=cancel_check)
    reference=None
    if fem_report.get('fem_verified'):
        if fem_report.get('design_sha256')!=canonical_physical_sha256(design):
            raise ValueError('FEM reference belongs to a different physical design/material')
        if fem_report.get('sample_fingerprint')!=sample_fingerprint(sample):
            raise ValueError('FEM reference uses different observation points, weights or domain')
        csv_file=Path(fem_report['reference_field_csv'])
        if fem_report.get('reference_field_sha256')!=hashlib.sha256(csv_file.read_bytes()).hexdigest():
            raise ValueError('FEM reference field CSV fingerprint mismatch')
        reference=read_field_csv(fem_report['reference_field_csv'])
        import numpy as np
        if np.shape(reference['points_mm'])!=np.shape(sample['points_mm']) or not np.allclose(reference['points_mm'],sample['points_mm'],atol=1e-9,rtol=0):
            raise ValueError('FEM reference coordinates do not match calibration sample')
    rows=[];selected=None
    for cells,order in ((1,2),(1,3),(2,2),(2,3)):
        if cancel_check:cancel_check()
        current=deepcopy(config)
        current['physics'].update(field_model='coupled',quadrature_order=order,cells_per_axis=cells)
        row={'cells_per_axis':cells,'quadrature_order':order,'physics':current['physics'],
            'model_fingerprint':model_fingerprint(current),'model_accuracy_passed':False}
        started=time.perf_counter()
        try:
            engine=make_field_engine(current,cancel_check=cancel_check)
            check=engine.preflight(design)
            row['preflight']=check
            if not check['valid']:raise ValueError('; '.join(check['violations']))
            fields=engine.field(design,sample['points_mm'])
            row.update(metrics=field_metrics(fields,sample['weights']),engine_statistics=dict(engine.statistics),status='python_completed')
            if reference is not None:
                comparison=compare_fem_fields(fields,reference['fields_T'],sample,fem_report.get('settings'))
                row.update(comparison=comparison,model_accuracy_passed=comparison['accuracy_passed'])
                if selected is None and row['model_accuracy_passed']:selected=deepcopy(current['physics'])
        except Exception as error:row.update(status='failed_or_resource_limited',error=str(error))
        row['elapsed_seconds']=time.perf_counter()-started;rows.append(row)
        if event_callback:event_callback({'event':'calibration_profile','profile':row})
        save_json(folder/'精度标定.json',{'profiles':rows,'selected_physics':selected,'fem':fem_report})
        # Once a profile passes the independent FEM gate, higher-cost profiles
        # are not needed to choose the first approved precision level.
        if selected is not None:break
    report={'schema':'v6-physics-calibration-1','design_sha256':canonical_physical_sha256(design),
        'design':design.to_dict(),'sample':sample,'profiles':rows,'selected_physics':selected,
        'fem_verified':bool(fem_report.get('fem_verified')),'model_accuracy_passed':selected is not None,
        'status':'passed' if selected else 'precision_not_accepted','fem':fem_report,
        'scope':'Calibration on this design and domain; final optimized candidate requires separate FEM acceptance'}
    save_json(folder/'精度标定.json',report)
    return report
