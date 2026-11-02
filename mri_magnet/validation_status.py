"""Keep training qualification separate from independent final acceptance."""
from .objectives import check_performance


def validation_status(training_metrics, dense_metrics, training_requirements, requirements):
    training = check_performance(training_metrics, training_requirements)
    dense = check_performance(dense_metrics, requirements)
    if dense['valid']:
        code, label = 'passed', '双门槛通过' if training['valid'] else '密集复核通过（训练未通过）'
    elif training['valid']:
        code, label = 'training_pass_dense_fail', '训练达标／复核未通过（待审查）'
    else:
        code, label = 'failed', '训练与复核均未通过'
    maximum = requirements.get('max_pp_norm_ppm')
    return {'code':code,'label':label,'training_pass':training['valid'],'dense_pass':dense['valid'],
        'field_pass':dense_metrics['mean_norm_T']+requirements.get('field_tolerance_T',0.) >= requirements.get('min_mean_norm_T',0.),
        'ppm_pass':maximum is None or dense_metrics['pp_norm_ppm']<=maximum,
        'ppm_rebound_vs_training':dense_metrics['pp_norm_ppm']-training_metrics['pp_norm_ppm'],
        'normalized_violation':dense['normalized_violation'],
        'ppm_excess':0. if maximum is None else max(0.,dense_metrics['pp_norm_ppm']-maximum),
        'minimum_field_deficit_mT':max(0.,requirements.get('min_mean_norm_T',0.)-dense_metrics['mean_norm_T'])*1000,
        'thresholds_relaxed':False}
