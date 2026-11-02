"""One domain contract for synthesis, optimization, previews and dense checks."""
import math
from functools import reduce
import numpy as np
from .sampling import sphere_samples, fundamental_sphere_samples, disk_samples, symmetry_profile


def make_sample(config, dense=False, designs=()):
    target, options = config['target'], config['sampling']
    prefix = 'validation' if dense else 'training'
    factor = 2 if options.get('preset') == 'standard' else 1
    nv, ns = (int(options[prefix+'_volume'])*factor, int(options[prefix+'_surface'])*factor)
    seed = int(config['seed']) + (100003 if dense else 71)
    profile = {'verified': False, 'used': False, 'reason': 'dense global validation' if dense else 'no common verified symmetry'}
    order = None
    # One physical point set across all islands; never optimize each on different quadrature.
    if designs and not dense and options.get('common_symmetry', True):
        order = reduce(math.gcd, (r.count for design in designs for r in design.rings))
        profiles = [symmetry_profile(d, rotation_order=order, reflection_phase_deg=-90.) for d in designs]
        valid = all(p['verified'] for p in profiles)
        # A noncentral disk does not have z reflection; its in-plane symmetry is still checked explicitly.
        profile = {'verified': valid, 'used': valid, 'rotation_order': order, 'meridian_plane_deg': -90.,
                   'design_profiles': profiles, 'reason': 'exact listed-model symmetry' if valid else 'fallback to global samples'}
        if not valid: order = None
    if target['geometry'] == 'disk':
        sample = disk_samples(target['diameter_mm'], nv, ns, seed, z_mm=target.get('z_mm', 0.), rotation_order=order)
    elif order:
        sample = fundamental_sphere_samples(target['diameter_mm'], nv, ns, seed, order, -90.)
    else:
        sample = sphere_samples(target['diameter_mm'], nv, ns, seed)
    sample['workbench_symmetry'] = profile
    return sample


def plot_samples(training, validation, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from .layout import _configure_font
    _configure_font()
    fig = plt.figure(figsize=(12, 5))
    for index, (sample, title) in enumerate(((training, '训练点'), (validation, '独立密集复核点'))):
        ax = fig.add_subplot(1, 2, index+1, projection='3d')
        pts = np.asarray(sample['points_mm']); weights = np.asarray(sample['weights'])
        ax.scatter(*pts[weights > 0].T, s=3, alpha=.5, color='#2672bc', label='积分点')
        ax.scatter(*pts[weights == 0].T, s=5, alpha=.5, color='#d78132', label='边界/特征点')
        ax.set(xlabel='x / mm', ylabel='y / mm', zlabel='z / mm', title=title+' · %d 点'%len(pts))
        ax.legend(fontsize=8); ax.set_box_aspect((1, 1, 1))
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)
