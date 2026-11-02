"""Persistent six-step workbench, independent of any source arrangement in de novo mode."""
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from pathlib import Path
import json
import math
import os
import runpy
import shutil
import hashlib
import threading
import time
import numpy as np

from .geometry import Design, validate_design, load_source
from .manufacturing import radial_packing_bounds, axial_packing_bounds, audit_manufacturing, _configuration, _bed_fit, make_manufacturing_seed, column_partition_check
from .serialization import save_json, jsonable
from .workbench_config import default_config, merge_config, validate_spec
from .workbench_domains import make_sample, plot_samples
from .objectives import check_performance, feasibility_score
from .identity import canonical_physical_sha256
from .config_schema import canonicalize_config

STEP_NAMES = ('模式与指标', '包络与打印工艺', '拓扑合成与起点', '采样与对称域', '求解与演化监控', '密集复核与交付')


class CancelledError(BaseException):
    """Propagate through the numerical core without being mistaken for a bad design."""


def requirements_for(config):
    target = config['target']
    maximum=target.get('max_ppm')
    scoring=deepcopy(config.get('scoring',{'mode':'ppm'}))
    return {'min_mean_norm_T': float(target['min_field_mT'])*1e-3,
        'max_pp_norm_ppm': None if maximum is None else float(maximum), 'field_tolerance_T': 0.,
        'field_scale_T': float(target['min_field_mT'])*1e-3,
        'ppm_scale': max(float(maximum if maximum is not None else scoring.get('ppm_scale',1000.)), 1.),
        'scoring':scoring,
        'rule': 'absolute hard limits, then balanced field/ppm scoring' if scoring.get('mode')=='balanced' else 'absolute user limits; strict feasibility then minimize ppm'}


def envelope_for(config):
    e = config['envelope']
    return {'bore_radius_mm': float(e['protected_bore_diameter_mm'])/2,
            'outer_radius_mm': float(e['max_outer_diameter_mm'])/2,
            'half_length_mm': float(e['max_length_mm'])/2,
            'min_gap_mm': 0., 'tolerance_mm': 1e-6, 'preserve_exact_bore': False}


def solver_settings_for(config, stop_file=''):
    """One settings adapter for the displayed budget plan and actual scheduler."""
    s=config['solver']
    return {'seed':config['seed'],'max_evaluations':s['max_evaluations'],'popsize':3,'maxiter':s['maxiter'],
        'optimization_options':{'population_size':s['population_size'],'initial_local_fraction':s['initial_local_fraction'],
            'initial_scale_mode':'physical_mm','initial_sigma_mm':s['initial_sigma_mm'],
            'local_refine_evaluations':s['local_refine_evaluations'],'pair_refine_fraction':s['pair_refine_fraction'],
            'pair_max_trials':s['pair_max_trials'],'active_set_evaluations':s.get('active_set_evaluations',240),
            'active_set_trust_radius':s.get('active_set_trust_radius',.03),'active_set_fd_step':s.get('active_set_fd_step',.002)},
        'islands':{'max_workers':s['max_workers'],'halving_factor':s['halving_factor'],
            'initial_budget_fraction':s['initial_budget_fraction'],'min_round_evaluations':2*s['population_size'],
            'graceful_stop':True,'stop_file':str(stop_file),'budget_policy':s.get('budget_policy','strict')}}


def preflight(config):
    """No field calls and no source IO. Bounds are necessary, not a completed design audit."""
    started = time.perf_counter()
    errors = validate_spec(config)
    result = {'valid': False, 'violations': errors, 'field_evaluations': 0}
    if errors: return result
    try:
        limits = envelope_for(config); process = _configuration(config['manufacturing'])
        if limits['outer_radius_mm'] <= limits['bore_radius_mm']: raise ValueError('最大外径必须大于保护内孔直径')
        cube = float(config['magnet']['cube_mm'])
        packing = radial_packing_bounds(cube, process, limits)
        if packing['maximum_radial_rings'] < 1: errors.append('包络内放不下带端部余量的一个磁槽')
        pocket = packing['pocket_edge_mm']; base = process['base_thickness_mm'] if process.get('enabled', True) else 0.
        gap = process['min_axial_module_gap_mm'] if process.get('enabled', True) else 0.
        axial=axial_packing_bounds(cube,process,limits)
        zmin,zmax=axial['minimum_abs_magnet_center_z_mm'],axial['maximum_abs_magnet_center_z_mm']
        dmin=axial['minimum_same_side_center_pitch_mm'];max_pairs=axial['maximum_layer_pairs']
        if max_pairs < 1: errors.append('轴向包络容不下一对中心镜像模块（含两块底板）')
        bed = _bed_fit(process, limits, pocket+base)
        if process.get('enabled', True) and not bed['fits']: errors.append('打印扇区连同床边预留超出打印床，需增加分块数或修改尺寸')
        target = config['target']; radius = target['diameter_mm']/2
        if radius > limits['bore_radius_mm']: errors.append('主评价域直径超出保护内孔，不能保证无源成像区域')
        if target['geometry'] == 'disk' and abs(target.get('z_mm', 0.)) > limits['half_length_mm']: errors.append('FOV 平面超出轴向包络')
        if target['geometry'] == 'sphere' and radius > limits['half_length_mm']: errors.append('DSV 超出轴向包络')
        aux = config['auxiliary_cylinder']
        if aux.get('enabled') and (aux['diameter_mm']/2 > limits['bore_radius_mm'] or aux['length_mm']/2 > limits['half_length_mm']): errors.append('辅助圆柱域超出无源保护包络')
        pairs = config['de_novo']['pair_counts']; columns = config['de_novo']['column_counts']
        if config['mode'] == 'de_novo':
            if not pairs or any(isinstance(p, bool) or int(p) != p or p<1 for p in pairs): errors.append('De Novo 层对数必须为正整数，并受实际轴向容纳上限限制')
            partitions=[column_partition_check(c,process) for c in columns]
            if not columns or any(not p['valid'] for p in partitions):
                errors.append('周向块数必须为不小于4的偶数；关闭跨缝磁槽时还须整除扇区数' if process.get('allow_split_pocket_seams') else '周向块数必须满足扇区整除规则')
            if not any(p <= max_pairs for p in pairs): errors.append('所选层对数全部超出轴向可容纳上限')
            eligible = [c for c,p in zip(columns,partitions) if p['valid']]
            if not eligible: errors.append('没有满足当前拼缝策略的周向列数')
            if eligible and pairs and 2*min(pairs)*min(eligible) > process['max_magnet_count']: errors.append('数量预算不足以让每个镜像层至少保留一个磁环')
            if not 3 <= int(config['de_novo']['seed_count']) <= 5: errors.append('拓扑簇数量应为 3～5')
        count = packing['maximum_radial_rings']
        slack = packing['maximum_magnet_center_radius_mm']-packing['minimum_magnet_center_radius_mm']-max(0,count-1)*packing['required_radial_center_pitch_mm']
        reserved = bed['reserved_bbox_mm']; dimensions = bed['printer_bed_mm']
        result.update(valid=not errors, packing=packing, max_radial_rings=count,
            radial_free_slack_mm=max(0., slack), radial_slack_by_count_mm={str(n): packing['maximum_magnet_center_radius_mm']-packing['minimum_magnet_center_radius_mm']-(n-1)*packing['required_radial_center_pitch_mm'] for n in range(1,count+1)},
            max_axial_pairs=max_pairs, axial_packing=axial, min_positive_center_z_mm=zmin, max_positive_center_z_mm=zmax,
            min_axial_center_pitch_mm=dmin, bed_fit=bed,
            best_bed_rotation_deg=bed['rotation_deg'], bed_bbox_occupancy_percent=100*reserved[0]*reserved[1]/(dimensions[0]*dimensions[1]),
            bed_material_area_percent=100*math.pi*(limits['outer_radius_mm']**2-limits['bore_radius_mm']**2)/process['sectors_per_layer']/(dimensions[0]*dimensions[1]),
            scope='necessary packing bounds; each actual design still requires full 3D/XY OBB manufacturing audit')
    except (ValueError, TypeError, KeyError, OverflowError) as error:
        errors.append(str(error))
    result['elapsed_seconds'] = time.perf_counter()-started
    return result


@dataclass
class ManufacturingChecker:
    process: dict
    limits: dict

    def __call__(self, design):
        nominal = validate_design(design, self.limits)
        audit = audit_manufacturing(design, self.process, self.limits)
        return {'valid': nominal['valid'] and audit['valid'], 'violations': nominal['violations']+audit['violations'], 'manufacturing': audit, 'geometry': nominal}


@dataclass
class FieldEvaluator:
    config: dict
    sample: dict
    cancel_file: str = ''
    graceful_stop: bool = False
    engine: object = field(default=None, init=False, repr=False)

    def __getstate__(self):
        state = dict(self.__dict__); state['engine'] = None
        return state

    def __call__(self, design):
        return self.evaluate_with_field(design)[0]

    def evaluate_with_field(self, design):
        if not self.graceful_stop and self.cancel_file and Path(self.cancel_file).exists(): raise CancelledError('用户停止运行')
        from .field import FieldEngine
        from .metrics import evaluate_design
        if self.engine is None: self.engine = FieldEngine(self.config['chunk_points'], self.config['polarization_model'])
        return evaluate_design(design, self.engine, self.sample, self.config['density_kg_m3'])

    def field_values(self,design):return self.evaluate_with_field(design)[1]


class Workbench:
    def __init__(self, project_dir, config_path=None):
        self.root = Path(project_dir).resolve(); self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock(); self.state_path = self.root/'工作台状态.json'
        if self.state_path.exists():
            self.data = json.loads(self.state_path.read_text(encoding='utf-8-sig'))
            self.data['config']=canonicalize_config(self.data['config'],validate=False)
        else:
            defaults=default_config()
            parameters=Path(__file__).resolve().parents[1]/'parameters.py'
            if parameters.is_file():
                defaults=canonicalize_config(runpy.run_path(str(parameters)).get('WORKBENCH_CONFIG',{}),base=defaults)
            self.data = {'config': defaults, 'steps': [{'number': i+1, 'name': n, 'status': 'pending', 'message': '待运行'} for i,n in enumerate(STEP_NAMES)],
                'seeds': [], 'selected_seed_ids': [], 'artifacts': {}, 'job': {}, 'run_dir': None, 'error': None}
        self.cancel_file = self.root/'停止请求.flag'
        if config_path:
            self.update_config(canonicalize_config(json.loads(Path(config_path).read_text(encoding='utf-8-sig'))))
        self._save()

    @property
    def config(self): return self.data['config']

    def _save(self):
        with self._lock:
            temp = self.state_path.with_suffix('.tmp.json')
            save_json(temp, self.data); os.replace(temp, self.state_path)

    def state(self):
        with self._lock: result=deepcopy(jsonable(self.data))
        self._complete_view_state(result)
        return result

    def analyze(self, design=None, source_label='', analysis_options=None):
        """Separate on-demand diagnostics; keep Step5/6 acceptance immutable."""
        from .field_analysis import analyze_design
        from .identity import canonical_physical_sha256
        with self._lock:
            config=deepcopy(self.config)
            if design is None:
                row=self.data.get('best') or self.data.get('current_best')
                if not row:
                    selected=self._selected();row=selected[0] if selected else None
                design=deepcopy(row.get('design')) if isinstance(row,dict) else None
            if not self._valid_view_design(design) or not design.get('rings') or design.get('preview_only'):
                raise ValueError('请先生成并选择真实磁体方案；初步几何示意不参与工程分析')
            value=Design.from_dict(design)
            if len(value.rings)>400 or value.magnet_count>10000:raise ValueError('按需分析上限为400环、10000块')
            if not math.isfinite(value.remanence_T) or value.remanence_T<=0 or not math.isfinite(value.reference_mur) or value.reference_mur<1:
                raise ValueError('分析要求有限正剩磁，回复相对磁导率不小于1')
            material=config['magnet']
            if any(not math.isclose(float(getattr(value,key)),float(material[key]),rel_tol=1e-10,abs_tol=1e-10) for key in ('cube_mm','remanence_T','reference_mur')):
                raise ValueError('所选方案材料与当前工程配置不一致；请先保存一致的配置')
            check=validate_design(value,envelope_for(config))
            if not check['valid']:raise ValueError('分析几何不合法: '+'；'.join(check['violations'][:6]))
            errors=validate_spec(config)
            if errors:raise ValueError('；'.join(errors))
            identity=json.dumps(config,sort_keys=True,allow_nan=False)
        stamp=datetime.now(timezone(timedelta(hours=8))).strftime('%Y%m%d_%H%M%S_%f')
        folder=self.root/'工程分析'/(stamp+'_'+canonical_physical_sha256(value)[:10])
        report=analyze_design(value,config,folder,analysis_options,source_label)
        report['manufacturing_audit']=audit_manufacturing(value,config['manufacturing'],envelope_for(config))
        report['artifacts']={key:self._relative(folder/name) for key,name in (
            ('report','工程分析.json'),('overview','工程分析速览.md'),('harmonics','harmonics.png')) if (folder/name).is_file()}
        with self._lock:
            if json.dumps(self.config,sort_keys=True,allow_nan=False)!=identity:
                raise RuntimeError('分析期间配置已改变；结果文件已保留，但不作为当前方案显示')
            self.data['analysis']=report
            save_json(folder/'工程分析.json',report)
            self._save()
        return report

    def _project_file(self, value):
        if not isinstance(value,str) or not value:return None
        try:
            path=Path(value);path=path if path.is_absolute() else self.root/path
            path=path.resolve();path.relative_to(self.root)
            return path if path.is_file() else None
        except (ValueError,OSError):return None

    def _view_json(self, path):
        """Read-only view cache, invalidated by file replacement or modification."""
        if path is None:return None
        try:
            stat=path.stat();raw=path.read_bytes()
            # Keep parsed JSON cached, but detect coarse Windows/cloud-drive
            # timestamps and same-size file replacements as well.
            key=(stat.st_mtime_ns,stat.st_ctime_ns,stat.st_size,hashlib.sha256(raw).digest())
            cache=getattr(self,'_view_json_cache',None)
            if cache is None:self._view_json_cache=cache={}
            cached=cache.get(str(path))
            if cached is not None and cached[0]==key:return cached[1]
            value=json.loads(raw.decode('utf-8-sig'))
            cache[str(path)]=(key,value)
            if len(cache)>64:cache.pop(next(iter(cache)),None)
            return value
        except (OSError,ValueError,TypeError):return None

    @staticmethod
    def _valid_view_design(value):
        """Validate render data only; never perform a field or manufacturing solve."""
        if not isinstance(value,dict) or not isinstance(value.get('rings'),(list,tuple)):return False
        try:
            cube=float(value.get('cube_mm',12.))
            if not math.isfinite(cube) or cube<=0:return False
            for ring in value['rings']:
                if not isinstance(ring,dict) or ring.get('polarity') not in ('out','in','down','up'):return False
                count=ring.get('count')
                if isinstance(count,bool) or not isinstance(count,(int,float)) or count<1 or int(count)!=count:return False
                for name in ('radius_mm','z_mm','phase_deg'):
                    if not math.isfinite(float(ring[name])):return False
                if float(ring['radius_mm'])<=0:return False
            return True
        except (ValueError,TypeError,KeyError,OverflowError):return False

    def _complete_candidate_views(self, result):
        """Attach the exact archived design to its own layout, without archive writes."""
        artifacts=result.get('artifacts') or {}
        manifest=self._project_file(artifacts.get('manifest'))
        archive=self._view_json(manifest)
        if not isinstance(archive,dict):archive={}

        def archive_file(value):
            if not isinstance(value,str) or not value:return None
            path=Path(value)
            if path.is_absolute():return self._project_file(value)
            local=self._project_file(str(manifest.parent/path)) if manifest else None
            return local or self._project_file(value)

        candidates=[]
        for row in archive.get('candidates',[]) or []:
            if not isinstance(row,dict):continue
            layout=archive_file(row.get('layout_file'))
            if layout is None:continue
            item={'path':self._relative(layout),'label':row.get('archive_label',row.get('candidate_id','候选方案')),
                'candidate_id':row.get('candidate_id'),'metrics':deepcopy(row.get('metrics'))}
            design_file=archive_file(row.get('design_file'))
            design=self._view_json(design_file) if design_file else None
            if not self._valid_view_design(design):design=row.get('design')
            if not self._valid_view_design(design):
                design_file=self._project_file(str(layout.parent/'设计.json'))
                design=self._view_json(design_file)
            if self._valid_view_design(design):item['design']=deepcopy(design)
            if design_file:item['design_file']=self._relative(design_file)
            candidates.append(item)
        if candidates:artifacts['candidate_layouts']=candidates
        else:candidates=[row for row in artifacts.get('candidate_layouts',[]) if isinstance(row,dict)]
        result['artifacts']=artifacts

        def find(record):
            identity=record.get('candidate_id')
            named=[row for row in candidates if identity and row.get('candidate_id')==identity]
            value=record.get('layout_file') or record.get('path') or record.get('layout')
            path=self._project_file(value)
            if path is None:path=archive_file(value)
            located=[row for row in candidates if path and row.get('path')==self._relative(path)]
            if len(named)==1 and len(located)==1 and named[0] is not located[0]:return None
            return named[0] if len(named)==1 else located[0] if len(located)==1 else None

        recommendations=result.get('pareto_recommendations')
        if not recommendations:
            recommendations=deepcopy(archive.get('pareto_recommendations') or [])
            for row in recommendations:
                if not isinstance(row,dict):continue
                for name in ('layout_file','class_file','java_file'):
                    path=archive_file(row.get(name))
                    if path:row[name]=self._relative(path)
            if recommendations:result['pareto_recommendations']=recommendations
        for row in recommendations or []:
            if not isinstance(row,dict):continue
            match=find(row)
            if match and self._valid_view_design(match.get('design')):
                row['design']=deepcopy(match['design'])
                if match.get('design_file'):row['design_file']=match['design_file']
        best=result.get('best')
        if isinstance(best,dict):
            match=find(best)
            if match and self._valid_view_design(match.get('design')):best['design']=deepcopy(match['design'])

    def _complete_view_state(self, result):
        """View-only compatibility for old saved runs; numerical archives stay immutable."""
        relative=result.get('run_dir');run=self.root/relative if relative else None
        evo=result.get('evolution') or {}
        solver_step=result.get('steps',[{}]*5)[4] if len(result.get('steps',[]))>=5 else {}
        if run and solver_step.get('status') in ('passed','stopped_with_result','running'):
            frames=evo.get('frames')
            if not frames:
                # Older v4 manifests omitted frames but the real standalone
                # timeline retained their metadata. Decode JSON, never script.
                timeline=self._project_file(str(run/'演化时间轴.html'))
                if timeline:
                    key=(str(timeline),timeline.stat().st_mtime_ns)
                    cached=getattr(self,'_legacy_frames_cache',None)
                    if not cached or cached[0]!=key:
                        text=timeline.read_text(encoding='utf-8-sig');decoded=[]
                        if 'const frames=' in text:
                            try:decoded=json.JSONDecoder().raw_decode(text.split('const frames=',1)[1])[0]
                            except (ValueError,TypeError):pass
                        self._legacy_frames_cache=(key,decoded if isinstance(decoded,list) else [])
                    frames=self._legacy_frames_cache[1]
            normalized=[]
            for frame in frames or []:
                row=dict(frame) if isinstance(frame,dict) else {'image':frame}
                name=row.get('image') or row.get('path')
                if not isinstance(name,str):continue
                path=self._project_file(name) or self._project_file(str(run/name))
                if path:
                    row['image']=self._relative(path);normalized.append(row)
            result['evolution']=dict(evo,frames=normalized,frames_count=len(normalized),frames_path_base='project')
        if not result.get('island_statuses'):
            islands={}
            known={s['id']:s for s in result.get('seeds',[]) if isinstance(s,dict) and 'id' in s}
            chosen=[known[i] for i in result.get('selected_seed_ids',[]) if i in known]
            names={i+1:s.get('style',s['id']) for i,s in enumerate(chosen)}
            for round_info in (result.get('optimization_metadata') or {}).get('island_rounds',[]):
                for row in round_info.get('islands',[]):
                    index=row['island'];m=row.get('best_metrics') or {}
                    islands[index]={'island':index,'name':names.get(index,'岛 %d'%index),
                        'round':round_info['round'],'status':'eliminated' if index in round_info.get('eliminated_islands',[]) else 'completed',
                        'stage':'complete','evaluations':row.get('field_evaluations',0),
                        'allocation':row.get('allocation'),'best_mean_mT':m.get('mean_norm_T',0)*1000,'best_ppm':m.get('pp_norm_ppm')}
            if islands:result['island_statuses']=list(islands.values())
        latest={}
        for entry in (result.get('optimization_metadata') or {}).get('runs',[]):
            index=entry.get('topology');number=entry.get('island_round',0)
            if index not in latest or number>=latest[index].get('island_round',0):latest[index]=entry
        for island in result.get('island_statuses',[]):
            metadata=latest.get(island.get('island'),{}).get('metadata',{})
            rounds=metadata.get('rounds') or []
            if rounds:
                local=rounds[-1].get('local_refinement') or {};lp=rounds[-1].get('active_set_polish') or {}
                island['refinement']={'coordinate_axes_visited':len(local.get('coordinate_axes_visited') or []),
                    'coordinate_sweeps_completed':local.get('coordinate_sweeps_completed',0),
                    'lp_passes':lp.get('linearization_passes',0),'lp_accepted_steps':lp.get('accepted_steps',0),
                    'lp_state':lp.get('state'),'lp_enabled':lp.get('enabled',False)}
        self._complete_candidate_views(result)

    def solver_preflight(self):
        from .solver_budget import plan_island_budgets
        selected=self._selected()
        if not selected:
            result={'valid':False,'status':'pending','policy':self.config['solver'].get('budget_policy','strict'),
                'violations':[],'warnings':['生成并选择起点后，才能按各岛实际自由维数核对预算'],
                'per_island':[],'rounds':[],'suggested_solver':{},'field_evaluations':0}
        else:
            result=plan_island_budgets([Design.from_dict(s['design']) for s in selected],
                self.config['manufacturing'],envelope_for(self.config),self.config['search'],
                solver_settings_for(self.config,self.cancel_file),requirements_for(self.config))
            for row,seed in zip(result.get('per_island',[]),selected):row['name']=seed.get('style',seed['id'])
        self.data['solver_preflight']=result;self._save();return result

    def apply_budget_suggestion(self):
        plan=self.solver_preflight();suggested=plan.get('suggested_solver') or {}
        if not suggested:raise ValueError('请先生成并选择起点，再计算预算建议')
        changes={key:value for key,value in suggested.items() if key in
            ('max_evaluations','population_size','local_refine_evaluations','active_set_evaluations','pair_refine_fraction')}
        changes['budget_policy']='strict'
        self.update_config({'solver':changes})
        return self.solver_preflight()

    def load_preset(self, preset_id):
        if preset_id=='default_20mm':cfg=default_config()
        elif preset_id=='engineering_100mm':
            path=Path(__file__).resolve().parents[1]/'工程案例_100mmDSV.json'
            cfg=canonicalize_config(json.loads(path.read_text(encoding='utf-8-sig')))
        else:raise ValueError('未知工程预设')
        self.update_config(cfg)
        return {'id':preset_id,'config':deepcopy(self.config),'message':'已加载预设；已有数值结果保留在原归档目录，未自动启动求解'}

    def _status(self, number, status, message, **values):
        with self._lock: self.data['steps'][number-1].update(status=status, message=message, **values)
        self._save()

    def _invalidate(self, first):
        for step in self.data['steps'][first-1:]:
            step.update(status='pending', message='输入已修改，需重新运行'); step.pop('preview', None)
        self.data['artifacts'] = {}; self.data['error'] = None
        self.data.pop('best',None);self.data.pop('current_best',None)
        self.data.pop('pareto_recommendations',None)
        self.data.pop('analysis',None)
        if first<=2:self.data.pop('preflight',None)
        if first<=5:
            for key in ('solver_preflight','evolution','optimization_metadata','island_statuses'):self.data.pop(key,None)
        if first<=5:self.data['job']={}

    def update_config(self, config):
        updated = canonicalize_config(config,base=self.config,validate=False)
        if updated != self.config:
            changed={key for key in updated if updated.get(key)!=self.config.get(key)}
            if changed=={'analysis'}:
                self.data['config']=updated;self.data.pop('analysis',None)
                self._save();return self.state()
            first=6
            if changed & {'mode','source_geometry','target'}: first=1
            elif changed & {'envelope','magnet','manufacturing'}: first=2
            elif changed & {'de_novo','search','density_kg_m3','polarization_model','chunk_points','seed'}: first=3
            elif 'sampling' in changed: first=4
            elif changed & {'solver','evolution','scoring'}: first=5
            self.data['config'] = updated
            if first in (4,5) and self._has_solver_result():self._fork_run()
            self._invalidate(first)
            if first<=3: self.data.update(seeds=[], selected_seed_ids=[], run_dir=None)
        self._save(); return self.state()

    def preflight(self):
        result = preflight(self.config); self.data['preflight'] = result
        self._status(2, 'passed' if result['valid'] else 'error', '包络必要条件通过；实际种子仍需完整审查' if result['valid'] else '；'.join(result['violations']))
        return result

    def _relative(self, path): return Path(path).resolve().relative_to(self.root).as_posix()

    def _relocate_references(self, old, new):
        def visit(value):
            if isinstance(value,dict):return {key:visit(item) for key,item in value.items()}
            if isinstance(value,list):return [visit(item) for item in value]
            if isinstance(value,str) and (value==old or value.startswith(old+'/')):return new+value[len(old):]
            return value
        self.data=visit(self.data)

    def _run_root(self):
        if not self.data.get('run_dir'):
            stamp = datetime.now(timezone(timedelta(hours=8))).strftime('%Y%m%d_%H%M%S_%f')
            self.data['run_dir'] = 'runs/'+stamp
        path = self.root/self.data['run_dir']; path.mkdir(parents=True, exist_ok=True)
        if not (path/'配置快照.json').exists(): save_json(path/'配置快照.json', self.config)
        snapshot=path/'详细数据'/'代码快照'
        if not snapshot.exists():
            snapshot.mkdir(parents=True)
            package=Path(__file__).resolve().parent;hashes={}
            for source in list(package.glob('*.py'))+list(package.glob('*.html'))+[package.parent/'parameters.py',package.parent/'requirements.txt']:
                if source.is_file():
                    destination=snapshot/source.name;shutil.copy2(source,destination)
                    hashes[source.name]=hashlib.sha256(destination.read_bytes()).hexdigest()
            save_json(snapshot.parent/'代码快照SHA256.json',hashes)
        return path

    def _has_solver_result(self):
        relative=self.data.get('run_dir')
        return bool(relative and (self.root/relative/'详细数据'/'优化结果.json').exists())

    def _fork_run(self):
        """A new solve gets a new archive; earlier completed results stay immutable."""
        old_relative=self.data.get('run_dir')
        if not old_relative:return
        old=self.root/old_relative
        self.data['run_dir']=None;new=self._run_root();new_relative=self.data['run_dir']
        for name in ('目标区域.svg','工艺包络预览.png','原始排布_RZ.png','采样点分布.png','工艺预检.json','拓扑种子簇.json'):
            if (old/name).is_file():shutil.copy2(old/name,new/name)
        for name in ('起点拓扑','详细数据/正向合成'):
            if (old/name).is_dir():shutil.copytree(old/name,new/name,dirs_exist_ok=True)
        for name in ('训练采样.json','复核采样.json','源槽投影对照.json'):
            if (old/'详细数据'/name).is_file():shutil.copy2(old/'详细数据'/name,new/'详细数据'/name)
        self._relocate_references(old_relative,new_relative)
        self.data['steps'][4].pop('preview',None);self.data['artifacts']={};self.data['job']={}

    def select_seeds(self, ids):
        ids = list(dict.fromkeys(ids))
        known = {s['id'] for s in self.data['seeds']}
        if not 1 <= len(ids) <= 3 or not set(ids) <= known: raise ValueError('请选择 1～3 个已生成的合法拓扑')
        if ids != self.data['selected_seed_ids']:
            if self._has_solver_result():self._fork_run()
            self.data['selected_seed_ids'] = ids; self._invalidate(4)
        self._save(); return self.state()

    def cancel(self):
        self.cancel_file.write_text('cancel', encoding='utf-8')
        return {'requested': True, 'message': '将在下一求场边界停止并保存当前最优解；有结果时可继续第6步'}

    def _check_cancel(self):
        if self.cancel_file.exists(): raise CancelledError('已按用户请求停止')

    def run_step(self, number, **options):
        number = int(number)
        if not 1 <= number <= 6: raise ValueError('步骤号必须为 1～6')
        if number > 1 and any(s['status'] not in ('passed','stopped_with_result') for s in self.data['steps'][:number-1]): raise ValueError('请先完成前置步骤；输入修改后须重新校验')
        self.cancel_file.unlink(missing_ok=True)
        self.data['error'] = None
        if number < 6: self._invalidate(number+1)
        self._status(number, 'running', '运行中')
        try:
            result = getattr(self, '_step%d'%number)(**options)
            status='stopped_with_result' if result.get('state')=='stopped_with_result' else 'passed'
            self._status(number, status, result.get('message', '通过'), **({'preview': result['preview']} if result.get('preview') else {}))
            return result
        except CancelledError:
            self.data['error'] = '用户停止运行'; self._status(number, 'pending', '已停止，快照保留')
            return {'cancelled': True, 'message': '已停止'}
        except Exception as error:
            self.data['error'] = str(error); self._status(number, 'error', str(error)); raise

    def _step1(self, **unused):
        errors = validate_spec(self.config)
        if errors: raise ValueError('；'.join(errors))
        if self.config['mode'] == 'seeded' and not self.config.get('source_geometry'): raise ValueError('Seeded 模式需要指定源排布路径')
        self._run_root(); save_json(self._run_root()/'配置快照.json', self.config)
        target=self.config['target'];is_sphere=target['geometry']=='sphere'
        label='球形 DSV' if is_sphere else '平面圆形 FOV（z=%g mm）'%target['z_mm']
        path=self._run_root()/'目标区域.svg'
        body='<circle cx="210" cy="150" r="92" fill="#deedf8" stroke="#2874a5" stroke-width="2"/>'
        if is_sphere:body+='<ellipse cx="210" cy="150" rx="92" ry="28" fill="none" stroke="#6493b2" stroke-dasharray="5 4"/><ellipse cx="210" cy="150" rx="28" ry="92" fill="none" stroke="#6493b2"/>'
        body+='<path d="M90 150H330 M210 35V265" stroke="#7b8d9d" stroke-dasharray="4 4"/>'
        ppm_label='峰峰值：不设硬上限，采用均衡评价' if target.get('max_ppm') is None else '峰峰值 ≤ %g ppm'%target['max_ppm']
        body+='<text x="365" y="105">%s · 直径 %g mm</text><text x="365" y="155">平均 |B| ≥ %g mT</text><text x="365" y="205">%s</text>'%(label,target['diameter_mm'],target['min_field_mT'],ppm_label)
        path.write_text('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 850 300"><rect width="850" height="300" fill="white"/><g font-family="Microsoft YaHei,sans-serif" font-size="20" fill="#243e52">'+body+'</g></svg>',encoding='utf-8')
        return {'message': '指标已确认：'+label+'；场强硬下限，'+('ppm 不设硬上限，按所选目标优化' if target.get('max_ppm') is None else 'ppm 使用硬上限'),'preview':self._relative(path)}

    def _step2(self, **unused):
        result = self.preflight()
        if not result['valid']: raise ValueError('；'.join(result['violations']))
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle, Wedge
        from .layout import _configure_font
        _configure_font(); fig, axes = plt.subplots(1,2,figsize=(11,4))
        p = result['packing']; cube = self.config['magnet']['cube_mm']; limits = envelope_for(self.config)
        ax = axes[0]
        for n in range(result['max_radial_rings']):
            r = p['minimum_magnet_center_radius_mm']+n*p['required_radial_center_pitch_mm']
            ax.add_patch(Rectangle((0,r-cube/2),cube,cube,color='#2e6ead'))
        ax.axhline(limits['bore_radius_mm'],color='#c44545',ls='--');ax.axhline(limits['outer_radius_mm'],color='#c44545',ls='--')
        ax.set(xlim=(-4,cube+4),ylim=(limits['bore_radius_mm']-5,limits['outer_radius_mm']+5),ylabel='R / mm',title='径向必要容纳上限 %d 环；余量 %.3f mm'%(result['max_radial_rings'],result['radial_free_slack_mm']))
        ax = axes[1]; bed = result['bed_fit']; angle = bed['rotation_deg']; sectors = self.config['manufacturing']['sectors_per_layer']
        a,b=math.radians(angle),math.radians(angle+360/sectors)
        angles=[a,b]+[q*math.pi/2 for q in range(-8,13) if a<=q*math.pi/2<=b]
        xy=np.array([(r*math.cos(t),r*math.sin(t)) for r in (limits['bore_radius_mm'],limits['outer_radius_mm']) for t in angles])
        bedx,bedy=bed['printer_bed_mm'][:2];center=np.array([bedx/2,bedy/2])-(xy.min(axis=0)+xy.max(axis=0))/2
        ax.add_patch(Rectangle((0,0),bedx,bedy,fill=False,edgecolor='#64748b',linestyle='--'))
        ax.add_patch(Wedge(center,limits['outer_radius_mm'],angle,angle+360/sectors,width=limits['outer_radius_mm']-limits['bore_radius_mm'],facecolor='#c5dcef',edgecolor='#2670aa'))
        ax.set(xlim=(-10,bedx+10),ylim=(-10,bedy+10),xlabel='打印床 x / mm',ylabel='打印床 y / mm',title='扇区最佳角 %.1f°；床面外包框占用 %.1f%%'%(angle,result['bed_bbox_occupancy_percent']));ax.set_aspect('equal')
        preview = self._run_root()/'工艺包络预览.png';fig.tight_layout();fig.savefig(preview,dpi=110);plt.close(fig)
        save_json(self._run_root()/'工艺预检.json',result)
        return dict(result,preview=self._relative(preview),message='必要容纳上限：径向%d环 / 轴向%d对；尚非完整拓扑审查'%(result['max_radial_rings'],result['max_axial_pairs']))

    def _step3(self, **unused):
        from .field import FieldEngine
        from .metrics import evaluate_design
        from .layout import render_rz_layout
        if self._has_solver_result():self._fork_run()
        cfg = self.config; sample = make_sample(cfg); root = self._run_root()
        engine = FieldEngine(cfg['chunk_points'],cfg['polarization_model'])
        if cfg['mode'] == 'de_novo':
            from .de_novo import synthesize
            result = synthesize(cfg,sample,engine=engine,output_dir=root/'详细数据'/'正向合成',log=self._log)
            seeds = result['seeds']; self.data['synthesis_metadata'] = result['metadata']
        else:
            from .workflow import _prepare_seed
            source_path = Path(cfg['source_geometry'])
            if not source_path.is_absolute(): source_path = Path(__file__).resolve().parents[1]/source_path
            source,_ = load_source(source_path)
            # Workbench magnet inputs are explicit, unlike the legacy optional source overrides.
            from dataclasses import replace
            source = replace(source, **cfg['magnet'])
            limits = envelope_for(cfg); search = dict(cfg['search']); search.setdefault('optimize_azimuth_count',False)
            search.setdefault('azimuth_count_candidates',[source.rings[0].count]); search['seed_subset_max_candidates'] = 36
            ref = engine.field(source,sample['points_mm'])
            def error(design):
                self._check_cancel(); return float(np.mean((engine.field(design,sample['points_mm'])-ref)**2))
            if len({r.layer_index for r in source.rings})%2 == 0:
                seed, audit = _prepare_seed(source,cfg['manufacturing'],limits,search,field_error=error)
            else: seed,audit = make_manufacturing_seed(source,cfg['manufacturing'],limits)
            full = audit_manufacturing(seed,cfg['manufacturing'],limits)
            if not full['valid']: raise ValueError('起点制造审查失败：'+'；'.join(full['violations']))
            metrics,_ = evaluate_design(seed,engine,sample,cfg['density_kg_m3'])
            seeds = [{'id':'seeded_01','style':'源槽组合评估 + PAVA 最小改动投影','design':seed.to_dict(),'metrics':metrics,'audit':full,'performance':check_performance(metrics,requirements_for(cfg)), 'projection':audit}]
            if any(search.get(flag,False) for flag in ('optimize_azimuth_count','optimize_radial_ring_count','optimize_magnetization')):
                from .topology import propose_topologies
                proposals=propose_topologies(source,seed,cfg['manufacturing'],limits,search,
                    np.random.default_rng(cfg['seed']),int(search.get('topology_limit',4)))
                for index,proposal in enumerate(proposals[1:],2):
                    self._check_cancel()
                    m,_=evaluate_design(proposal,engine,sample,cfg['density_kg_m3'])
                    a=audit_manufacturing(proposal,cfg['manufacturing'],limits)
                    seeds.append({'id':'seeded_%02d'%index,'style':'源排布拓扑候选 %d'%index,
                        'design':proposal.to_dict(),'metrics':m,'audit':a,'performance':check_performance(m,requirements_for(cfg))})
                topology_metadata=proposals.metadata
            else:topology_metadata={'returned_count':1,'reason':'all discrete topology switches disabled'}
            self.data['synthesis_metadata'] = {'mode':'seeded','source_file':str(source_path),'seed_selection':audit.get('seed_selection'),'seed_catalog_examined':audit.get('seed_catalog_examined'), 'source_design':source.to_dict()}
            self.data['synthesis_metadata']['topology_proposals']=topology_metadata
            render_rz_layout(source,root/'原始排布_RZ.png',title='源排布（投影前）')
            save_json(root/'详细数据'/'源槽投影对照.json', audit)
        if not seeds: raise ValueError('未找到通过制造审查的种子；请查看正向合成诊断或放宽指标/包络')
        for row in seeds:
            self._check_cancel(); design = Design.from_dict(row['design'])
            checked = audit_manufacturing(design,cfg['manufacturing'],envelope_for(cfg))
            if not checked['valid']: raise ValueError('生成器返回未通过完整审查的种子')
            row['audit'] = checked
            path = root/'起点拓扑'/str(row['id'])/'磁铁布局_RZ.png'
            source_data = self.data['synthesis_metadata'].get('source_design')
            render_rz_layout(design,path,title='%s · %.3f mT / %.3f ppm'%(row['style'],row['metrics']['mean_norm_T']*1000,row['metrics']['pp_norm_ppm']),manufacturing_audit=checked,reference_design=Design.from_dict(source_data) if source_data else None,reference_label='源排布')
            row['preview'] = self._relative(path)
        self.data['seeds'] = seeds
        ordered = sorted(seeds,key=lambda s:feasibility_score(s['metrics'],requirements_for(cfg)))
        self.data['selected_seed_ids'] = [s['id'] for s in ordered[:3]]
        self.solver_preflight()
        save_json(root/'拓扑种子簇.json', {'seeds':seeds,'metadata':self.data['synthesis_metadata']})
        return {'seeds':seeds,'message':'已生成 %d 个完整制造审查通过的起点；性能达标状态单独列出'%len(seeds),'preview':seeds[0]['preview']}

    def _selected(self):
        # Preserve selection order so topology_index is stable in monitor and archives.
        rows = {s['id']:s for s in self.data['seeds']}
        return [rows[i] for i in self.data['selected_seed_ids']]

    def _step4(self, **unused):
        from .metrics import evaluate_design
        from .field import FieldEngine
        selected = self._selected()
        if not selected: raise ValueError('请先选择起点')
        designs = [Design.from_dict(s['design']) for s in selected]
        training = make_sample(self.config,designs=designs); validation = make_sample(self.config,dense=True)
        root = self._run_root(); save_json(root/'详细数据'/'训练采样.json',training);save_json(root/'详细数据'/'复核采样.json',validation)
        engine = FieldEngine(self.config['chunk_points'],self.config['polarization_model'])
        for row, design in zip(selected,designs):
            row['training_metrics'] = evaluate_design(design,engine,training,self.config['density_kg_m3'])[0]
        self.data['sampling'] = {'training_count':len(training['points_mm']),'validation_count':len(validation['points_mm']), 'symmetry':training['workbench_symmetry'], 'geometry':self.config['target']['geometry']}
        path = root/'采样点分布.png';plot_samples(training,validation,path)
        return {'message':'训练%d点 / 独立复核%d点；%s'%(len(training['points_mm']),len(validation['points_mm']),'共同对称基本域已验证' if training['workbench_symmetry']['used'] else '使用全域采样'),'preview':self._relative(path),**self.data['sampling']}

    def _log(self, text):
        self.data['job']['message'] = str(text); self._check_cancel()

    def _step5(self, **unused):
        from .islands import run_islands
        from .evolution import EvolutionMonitor
        plan=self.solver_preflight()
        if not plan.get('valid'):raise ValueError('求解预算预检未通过：'+'；'.join(plan.get('violations') or plan.get('warnings') or ['请先选择起点']))
        if self._has_solver_result():self._fork_run()
        cfg = self.config; selected = self._selected(); root = self._run_root()
        designs = [Design.from_dict(s['design']) for s in selected]
        sample = make_sample(cfg,designs=designs); limits = envelope_for(cfg); requirements = requirements_for(cfg)
        reference = min(selected,key=lambda s:feasibility_score(s.get('training_metrics',s['metrics']),requirements))
        monitor = EvolutionMonitor(root,Design.from_dict(reference['design']),reference.get('training_metrics',reference['metrics']),requirements,cfg,
            reference_designs=[(d,s.get('training_metrics',s['metrics'])) for d,s in zip(designs,selected)])
        solver = cfg['solver']
        settings=solver_settings_for(cfg,self.cancel_file)
        self.data['current_best']=None
        self.data.pop('evolution',None)
        self.data['island_statuses']=[{'island':i+1,'name':s.get('style',s['id']),'round':1,'status':'queued','stage':'initialization','evaluations':0} for i,s in enumerate(selected)]
        self.data['job']={'max_evaluations':solver['max_evaluations'],'evaluations':0,'progress_percent':0.,'log_tail':[]}
        def event_callback(event):
            count=int(event.get('global_evaluations',0));index=event.get('topology_index')
            with self._lock:
                if isinstance(index,int) and 1<=index<=len(selected):
                    island=self.data['island_statuses'][index-1]
                    island.update(round=event.get('island_round',island.get('round',1)),
                        evaluations=int(event.get('field_evaluations',island.get('evaluations',0))))
                    if event.get('stage_name') or event.get('phase'):island['stage']=event.get('stage_name',event.get('phase'))
                    if event.get('island_status'):island['status']=event['island_status']
                    elif event.get('event') not in ('stage_end','final'):island['status']='running'
                    if event.get('allocation') is not None:island['allocation']=event['allocation']
                    if event.get('metrics'):
                        island.update(best_mean_mT=event['metrics']['mean_norm_T']*1000,best_ppm=event['metrics']['pp_norm_ppm'])
                self.data['job'].update(evaluations=count,stage=event.get('stage_name',event.get('phase','')),topology_index=index,
                    progress_percent=min(100.,100*count/solver['max_evaluations']),max_evaluations=solver['max_evaluations'],
                    topology_name=selected[index-1]['style'] if isinstance(index,int) and 1<=index<=len(selected) else '拓扑岛')
                if event.get('design') and event.get('metrics'):
                    best=self.data.get('current_best')
                    if best is None or feasibility_score(event['metrics'],requirements)<feasibility_score(best['metrics'],requirements):
                        self.data['current_best']=deepcopy(event)
                    best=self.data['current_best']
                    self.data['job'].update(best_mean_mT=best['metrics']['mean_norm_T']*1000,best_ppm=best['metrics']['pp_norm_ppm'],
                        metric_scope='training',best_pointset=best['metrics'].get('pointset'))
                logs=self.data['job'].setdefault('log_tail',[])
                logs.append('%s · 岛%s · 求场%d/%d · %s'%(event.get('event',''),index,count,solver['max_evaluations'],self.data['job']['stage']))
                del logs[:-80]
            monitor(event)
            if (root/'实时演化看板.png').exists(): self.data['steps'][4]['preview'] = self._relative(root/'实时演化看板.png')
            if monitor.frames:self.data['job']['latest_frame']=self._relative(root/monitor.frames[-1]['image'])
            now=time.monotonic()
            if now-getattr(self,'_last_progress_save',0.)>=1.:
                self._save();self._last_progress_save=now
        evaluator = FieldEvaluator(deepcopy(cfg),sample,str(self.cancel_file),graceful_stop=True)
        checker = ManufacturingChecker(deepcopy(cfg['manufacturing']),limits)
        try:
            result = run_islands(designs,cfg['manufacturing'],limits,cfg['search'],evaluator,checker,root/'详细数据'/'多岛优化',settings,requirements,event_callback=event_callback)
            best = result.get('diagnostic_best')
            if best: monitor(dict(best,event='final',global_evaluations=result['metadata']['field_evaluations']))
            used=result['metadata']['field_evaluations']
            self.data['job'].update(evaluations=used,progress_percent=min(100.,100*used/solver['max_evaluations']),
                status=result.get('state','complete'))
            if best:
                self.data['current_best']=deepcopy(best)
                self.data['job'].update(best_mean_mT=best['metrics']['mean_norm_T']*1000,best_ppm=best['metrics']['pp_norm_ppm'])
            save_json(root/'详细数据'/'优化结果.json',result)
            self.data['optimization_metadata'] = result['metadata']
            if result.get('records'):
                import matplotlib
                matplotlib.use('Agg')
                import matplotlib.pyplot as plt
                from .layout import _configure_font
                _configure_font();fig,axes=plt.subplots(2,1,figsize=(10,6),sharex=True)
                counts=[];ppms=[];fields=[];incumbent=None
                for row in sorted(result['records'],key=lambda r:r.get('global_evaluations',0)):
                    if incumbent is None or feasibility_score(row['metrics'],requirements)<feasibility_score(incumbent['metrics'],requirements):incumbent=row
                    counts.append(row.get('global_evaluations',0));ppms.append(incumbent['metrics']['pp_norm_ppm']);fields.append(incumbent['metrics']['mean_norm_T']*1000)
                axes[0].plot(counts,ppms,color='#337dba');axes[0].set(ylabel='当前推荐解 ppm',yscale='symlog')
                if cfg['target'].get('max_ppm') is not None:axes[0].axhline(cfg['target']['max_ppm'],color='#69a88c',ls='--',lw=.8)
                axes[1].plot(counts,fields,color='#337dba');axes[1].set(xlabel='累计搜索求场次数（按预算台账）',ylabel='当前最佳平均场 / mT');axes[1].axhline(cfg['target']['min_field_mT'],color='#69a88c',ls='--',lw=.8)
                for axis in axes:axis.grid(alpha=.15)
                fig.suptitle('真实搜索收敛记录 · 训练点集；独立密集复核另列')
                fig.tight_layout();fig.savefig(root/'优化收敛曲线.png',dpi=110);plt.close(fig)
                self.data['artifacts']['convergence']=self._relative(root/'优化收敛曲线.png')
        finally:
            evolution = monitor.finish(); save_json(root/'详细数据'/'演化索引.json',evolution)
            if (root/'优化收敛曲线.png').is_file():
                evolution['convergence_file']='优化收敛曲线.png';save_json(root/'详细数据'/'演化索引.json',evolution)
            self.data['evolution'] = evolution
            if evolution.get('gif_file'):
                (root/'00_演化过程.gif').write_bytes((root/evolution['gif_file']).read_bytes())
        self.data['artifacts'].update(live=self._relative(root/'实时演化看板.png') if (root/'实时演化看板.png').exists() else None,
            timeline=self._relative(root/'演化时间轴.html') if (root/'演化时间轴.html').exists() else None)
        state=result.get('state',result.get('metadata',{}).get('state'))
        if not result.get('records'):raise CancelledError('尚无可保存求场结果')
        stopped=state=='stopped_with_result'
        return {'state':'stopped_with_result' if stopped else 'complete',
            'message':('已停止并保存最优解，可进入第6步；' if stopped else '多岛优化结束；')+'已用 %d / %d 次搜索求场，待独立密集复核'%(result['metadata']['field_evaluations'],solver['max_evaluations']), 'preview':self.data['artifacts'].get('live'), 'metadata':result['metadata']}

    def export_current(self, compile=False, **unused):
        """Export an immutable training snapshot without changing a running solver."""
        from .layout import render_rz_layout
        from .comsol_export import export_comsol_model
        with self._lock:
            row=deepcopy(self.data.get('current_best'));cfg=deepcopy(self.config);seeds=deepcopy(self._selected())
        if not row or not row.get('design'):raise ValueError('尚无已求场的当前最优解；请先运行求解器')
        stamp=datetime.now(timezone(timedelta(hours=8))).strftime('%Y%m%d_%H%M%S_%f')
        folder=self.root/'当前解快照'/stamp;folder.mkdir(parents=True)
        design=Design.from_dict(row['design']);index=row.get('topology_index')
        reference=Design.from_dict(seeds[index-1]['design']) if isinstance(index,int) and 1<=index<=len(seeds) else None
        save_json(folder/'训练点当前解.json',row);save_json(folder/'配置快照.json',cfg)
        layout=folder/'磁铁布局_RZ.png'
        render_rz_layout(design,layout,title='当前搜索解（训练点；未密集复核） · %.6f mT / %.6f ppm'%(row['metrics']['mean_norm_T']*1000,row['metrics']['pp_norm_ppm']),reference_design=reference,reference_label='本岛起点')
        options=dict(cfg['deliverables'],compile=bool(compile),runtime_validate=False)
        model=export_comsol_model(design,folder,cfg['comsol_root'],options)
        files={'layout':self._relative(layout),'design':self._relative(folder/'训练点当前解.json'),'java':self._relative(model['java_file'])}
        if model.get('class_file'):files['class']=self._relative(model['class_file'])
        info={'status':'training_snapshot_not_dense_verified','message':'已导出当前搜索解；这是训练点指标，尚未密集复核或 FEM 验证',
            'artifacts':files,'metrics':row['metrics'],'model_status':model['status'],'dense_verified':False,'fem_verified':False}
        save_json(folder/'快照说明.json',info)
        if compile and not model.get('compiled'):raise RuntimeError('快照已保存，但CLASS编译失败：'+str(model.get('error')))
        return info

    def _step6(self, export_comsol=True, compile=None, **unused):
        from .metrics import evaluate_design
        from .field import FieldEngine
        from .validation import refine_sphere_extrema, refine_disk_extrema
        from .sampling import cylinder_samples
        from .workbench_archive import archive_run
        cfg = self.config; root = self._run_root(); req = requirements_for(cfg)
        save_json(root/'详细数据'/'导出请求.json',{'export_comsol':bool(export_comsol),'compile_override':compile,
            'configured_deliverables':cfg['deliverables'],'comsol_root':cfg['comsol_root']})
        result = json.loads((root/'详细数据'/'优化结果.json').read_text(encoding='utf-8'))
        from .objectives import select_validation_candidates
        rows = result['records']; ordered = sorted(rows,key=lambda r:feasibility_score(r['metrics'],req))
        # One representative per island, then global Top-K: keep diagnostics even when a dense gate fails.
        chosen=[]; keys=set()
        def add(row):
            key = canonical_physical_sha256(row['design'])
            if key not in keys and len(chosen)<cfg['max_validation_candidates']: chosen.append(row);keys.add(key)
        for row in select_validation_candidates(rows,req,limit=cfg['max_validation_candidates']): add(row)
        for index in range(1,len(self._selected())+1):
            group=[r for r in ordered if r.get('topology_index')==index]
            if group: add(group[0])
        for row in ordered: add(row)
        if not chosen: raise ValueError('没有有限求场结果可复核')
        engine = FieldEngine(cfg['chunk_points'],cfg['polarization_model'])
        sample = make_sample(cfg,dense=True); starts = cfg['sampling']['refine_starts']
        seeds = deepcopy(self._selected())
        for seed in seeds:
            self._check_cancel(); design = Design.from_dict(seed['design'])
            m,b = evaluate_design(design,engine,sample,cfg['density_kg_m3'])
            refine = refine_sphere_extrema if cfg['target']['geometry']=='sphere' else refine_disk_extrema
            seed['dense_metrics'] = refine(design,engine,sample,m,b,starts=starts)
            seed['validation_metrics'] = seed['dense_metrics']
        validated=[]; checker = ManufacturingChecker(cfg['manufacturing'],envelope_for(cfg))
        for index, row in enumerate(chosen):
            self._check_cancel(); current=deepcopy(row); design=Design.from_dict(current['design'])
            audit=checker(design)
            if not audit['valid']: raise ValueError('密集复核前的几何重审失败')
            metrics,B = evaluate_design(design,engine,sample,cfg['density_kg_m3'])
            refine = refine_sphere_extrema if cfg['target']['geometry']=='sphere' else refine_disk_extrema
            metrics = refine(design,engine,sample,metrics,B,starts=starts)
            refinement = metrics.get('extrema_refinement',{})
            current.update(training_metrics=current['metrics'],metrics=metrics,performance_check=check_performance(metrics,req),audit=audit,refinement=refinement)
            train_pass=check_performance(current['training_metrics'],req)['valid']; dense_pass=current['performance_check']['valid']
            passed_label='双门槛通过' if cfg['target'].get('max_ppm') is not None else '场强硬门槛通过'
            current['validation_status']={'label':passed_label if dense_pass else '训练达标 / 复核待裕度检查' if train_pass else '性能未达标（诊断方案）','dense_pass':dense_pass,'training_pass':train_pass,'retained_for_comparison':True}
            if cfg['auxiliary_cylinder']['enabled']:
                aux=cfg['auxiliary_cylinder']; cylinder=cylinder_samples(aux['diameter_mm'],aux['length_mm'],256,seed=cfg['seed']+991)
                cylinder.update(geometry='cylinder',measure='volume')
                current['auxiliary_metrics']=evaluate_design(design,engine,cylinder,cfg['density_kg_m3'])[0]
                current['auxiliary_scope']='辅助域采样结果，不替代主评价域，也未连续细化圆柱极值'
            validated.append(current);self.data['job'].update(message='密集复核 %d/%d'%(index+1,len(chosen)))
        save_json(root/'详细数据'/'密集复核.json',validated)
        save_json(root/'详细数据'/'起点密集复核.json',seeds)
        evolution=json.loads((root/'详细数据'/'演化索引.json').read_text(encoding='utf-8'))
        best = min(validated,key=lambda r:feasibility_score(r['metrics'],req))
        suffix='_最优%.2fmT_%.3fppm'%(best['metrics']['mean_norm_T']*1000,best['metrics']['pp_norm_ppm'])
        target=root.with_name(root.name.split('_最优')[0]+suffix)
        if target!=root:
            if target.parent.resolve() != (self.root/'runs').resolve(): raise ValueError('结果重命名路径超出工程目录')
            old=self.data['run_dir'];root.rename(target)
            self._relocate_references(old,self._relative(target));root=target
        archive=archive_run(root,cfg,seeds,validated,result,evolution,export_comsol=export_comsol,compile_class=compile)
        artifacts=archive.get('artifacts',{})
        if (root/'优化收敛曲线.png').is_file():artifacts['convergence']='优化收敛曲线.png'
        self.data['artifacts']={key:self._relative(root/value) if isinstance(value,str) and not Path(value).is_absolute() else value for key,value in artifacts.items()}
        recommendations=deepcopy(archive.get('pareto_recommendations',[]))
        for row in recommendations:
            for key in ('layout_file','folder','class_file','java_file'):
                if row.get(key) and not Path(row[key]).is_absolute():row[key]=self._relative(root/row[key])
        self.data['pareto_recommendations']=recommendations
        self.data['best']={'metrics':best['metrics'],'design':deepcopy(best['design']),'candidate_id':best['candidate_id'],'validation_status':best['validation_status'],'comsol_fem_verified':False}
        requested_compile=cfg['deliverables'].get('compile',True) if compile is None else bool(compile)
        if export_comsol and requested_compile:
            failed=[r for r in archive.get('candidates',[]) if r.get('model_export') and not r['model_export'].get('compiled')]
            if failed:raise RuntimeError('Python 复核结果已归档，但 %d 个 CLASS 编译失败；请查看导出日志和报告'%len(failed))
        return {'message':'复核与归档完成；'+best['validation_status']['label']+'；COMSOL FEM 尚未验证', 'preview':self.data['artifacts'].get('layout'), 'artifacts':self.data['artifacts']}
