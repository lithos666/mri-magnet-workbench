"""Optional bounded planar preview; its field work is separate from search."""
from pathlib import Path
import json,time,html,os
from collections import OrderedDict
import numpy as np
from .geometry import Design
from .field import FieldEngine
from .identity import canonical_physical_sha256
from .layout import _configure_font,layout_comparison_manifest
from .serialization import save_json,jsonable
from .objectives import check_performance,feasibility_score


class EvolutionMonitor:
    def __init__(self,public_dir,seed_design,seed_metrics,requirements,config,reference_designs=None):
        self.root=Path(public_dir).resolve();self.options=dict(config.get('evolution',{}))
        self.enabled=bool(self.options.get('enabled',False));self.error=None
        self.seed=seed_design;self.seed_metrics=seed_metrics;self.requirements=requirements
        from .parameterization import topology_identity
        references=reference_designs or [(seed_design,seed_metrics)]
        self.references={topology_identity(d):(d,m) for d,m in references}
        self.references_by_index={i+1:value for i,value in enumerate(references)}
        self.fixed_color_limit=max(max(float(m['pp_norm_ppm']) for _,m in references)/2,1e-5)
        self.events=[];self.frames=[];self.render_count=0;self.overhead_seconds=0.;self.last_render=-1e30
        target=config.get('target',{})
        self.region=target.get('geometry','disk')
        self.diameter=float(target.get('diameter_mm',config.get('fov_diameter_mm',20.)))
        self.z=float(target.get('z_mm',config.get('fov_z_mm',0.)));self.previous_line=None;self.incumbent=None
        if self.region not in ('sphere','disk'):raise ValueError('Unsupported preview geometry')
        self.preview_fields=OrderedDict();self.field_cache_hits=0
        self.max_frames=max(3,int(self.options.get('max_frames',24)))
        self.minimum_seconds=float(self.options.get('min_refresh_seconds',5.))
        self.engine=FieldEngine(config['chunk_points'],config['polarization_model'],prepared_cache_size=2)
        if self.enabled:
            self.folder=self.root/'详细数据'/'演化过程';self.folder.mkdir(parents=True,exist_ok=True)
            self.history_path=self.folder/'history_frames.jsonl'
            # Every preview point lies on the requested disk, including both lines.
            n=max(9,int(self.options.get('grid_points',21)));self.axis=np.linspace(-self.diameter/2,self.diameter/2,n)
            x,y=np.meshgrid(self.axis,self.axis);self.xy=np.c_[x.ravel(),y.ravel()]
            self.mask=np.sum(self.xy*self.xy,axis=1)<=(self.diameter/2)**2+1e-10
            self.map_points=np.c_[self.xy[self.mask],np.full(np.count_nonzero(self.mask),self.z)]
            self.line_x=np.linspace(-self.diameter/2,self.diameter/2,61)
            self.lines=np.vstack((np.c_[self.line_x,np.zeros(61),np.full(61,self.z)],
                np.c_[np.zeros(61),self.line_x,np.full(61,self.z)]))
            if self.region=='sphere':
                self.map_points=np.c_[self.xy[self.mask,0],np.zeros(np.count_nonzero(self.mask)),self.xy[self.mask,1]]
                self.lines=np.vstack((np.c_[np.zeros(61),np.zeros(61),self.line_x],
                    np.c_[self.line_x,np.zeros(61),np.zeros(61)]))
            self.points=np.vstack((self.map_points,self.lines))
            self.base_fields=None

    def _remember(self,event):
        selected={k:event[k] for k in ('event','phase','stage_name','topology_index','global_evaluations',
            'field_evaluations','generation','candidate_id','design','metrics','validation_status','population_metrics','population_scope') if k in event}
        selected['event_index']=len(self.events);self.events.append(selected)
        with self.history_path.open('a',encoding='utf-8') as stream:
            stream.write(json.dumps(jsonable(selected),ensure_ascii=False,allow_nan=False)+'\n')
        return selected

    def __call__(self,event):
        if not self.enabled or self.error:return
        started=time.perf_counter()
        try:
            event=self._remember(event)
            if not event.get('design') or not event.get('metrics'):return
            if event.get('event') in ('validation','final'):
                rendered=event
            else:
                if self.incumbent is None or feasibility_score(event['metrics'],self.requirements)<feasibility_score(self.incumbent['metrics'],self.requirements):self.incumbent=event
                rendered=dict(event,design=self.incumbent['design'],metrics=self.incumbent['metrics'],candidate_id=self.incumbent.get('candidate_id'),topology_index=self.incumbent.get('topology_index'))
            if self.render_count>=self.max_frames:return
            if self.render_count>=self.max_frames-1 and event.get('event')!='final':return
            forced=event.get('event') in ('baseline','stage_end','validation','final')
            if not forced and (len(self.frames)>=self.max_frames-2 or time.perf_counter()-self.last_render<self.minimum_seconds):return
            if event.get('event')=='generation' and int(event.get('generation',0))%max(1,int(self.options.get('generation_interval',1))):return
            self._render(rendered);self.last_render=time.perf_counter()
        except Exception as error:
            self.error=str(error)
            save_json(self.folder/'monitor_error.json',{'error':self.error,'search_continues':True})
        finally:self.overhead_seconds+=time.perf_counter()-started

    def _render(self,event):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
        _configure_font();design=Design.from_dict(event['design']);metrics=event['metrics']
        from .parameterization import topology_identity
        reference=self.references_by_index.get(event.get('topology_index')) or self.references.get(topology_identity(design))
        if reference and topology_identity(reference[0])!=topology_identity(design):reference=None
        if reference and reference[0] is not self.seed:
            self.seed,self.seed_metrics=reference;self.base_fields=None;self.previous_line=None
        def preview_field(current):
            key=canonical_physical_sha256(current,getattr(self.engine,'polarization_model','remanence'))
            if key in self.preview_fields:
                self.field_cache_hits+=1;values=self.preview_fields.pop(key);self.preview_fields[key]=values;return values
            values=self.engine.field(current,self.points);self.preview_fields[key]=values
            while len(self.preview_fields)>4:self.preview_fields.popitem(last=False)
            return values
        if self.base_fields is None:self.base_fields=preview_field(self.seed)
        fields=preview_field(design);norm=np.linalg.norm(fields,axis=1)
        mean=float(metrics['mean_norm_T']);ppm=(norm-mean)/mean*1e6
        grid=np.full((len(self.axis)**2,),np.nan);grid[self.mask]=ppm[:len(self.map_points)];grid=grid.reshape(len(self.axis),-1)
        fig,axes=plt.subplots(2,2,figsize=(16,9),gridspec_kw={'width_ratios':[1.25,1.]})
        ax=axes[0,0];comparison=layout_comparison_manifest(design,self.seed)
        for ring in self.seed.rings:
            ax.add_patch(Rectangle((ring.z_mm-self.seed.cube_mm/2,ring.radius_mm-self.seed.cube_mm/2),self.seed.cube_mm,self.seed.cube_mm,fill=False,edgecolor='#9ca3af',linestyle='--',linewidth=.65))
        palette={'out':'#aa2f36','in':'#1955b0','down':'#a56e27','up':'#22877a'}
        ref={(r.layer_index,r.slot_index):r for r in self.seed.rings}
        current_keys={(r.layer_index,r.slot_index) for r in design.rings}
        for ring in self.seed.rings:
            if (ring.layer_index,ring.slot_index) not in current_keys:
                ax.add_patch(Rectangle((ring.z_mm-self.seed.cube_mm/2,ring.radius_mm-self.seed.cube_mm/2),self.seed.cube_mm,self.seed.cube_mm,facecolor='#64748b',alpha=.12))
        for ring in design.rings:
            ax.add_patch(Rectangle((ring.z_mm-design.cube_mm/2,ring.radius_mm-design.cube_mm/2),design.cube_mm,design.cube_mm,facecolor=palette[ring.polarity],edgecolor='white',linewidth=.35))
            original=ref.get((ring.layer_index,ring.slot_index))
            if original and np.hypot(ring.z_mm-original.z_mm,ring.radius_mm-original.radius_mm)>.5:
                ax.annotate('',xy=(ring.z_mm,ring.radius_mm),xytext=(original.z_mm,original.radius_mm),arrowprops=dict(arrowstyle='->',color='#f59e0b',lw=.75))
                if self.region=='sphere':ax.annotate('%.1f mm'%np.hypot(ring.z_mm-original.z_mm,ring.radius_mm-original.radius_mm),(ring.z_mm,ring.radius_mm),fontsize=5,color='#986013')
        allrings=list(self.seed.rings)+list(design.rings)
        ax.set_xlim(max(r.z_mm for r in allrings)+14,min(r.z_mm for r in allrings)-14)
        ax.set_ylim(min(r.radius_mm for r in allrings)-14,max(r.radius_mm for r in allrings)+14)
        ax.axvline(0,color='#aaa',lw=.7,ls='--');ax.set(xlabel='Z / mm（正Z在左）',ylabel='R / mm',title='核心排布：灰虚线=打印起点；橙箭头=位移')
        ax.set_aspect('equal',adjustable='box')
        heat=axes[0,1];limit=self.fixed_color_limit
        cmap='RdBu_r'
        if self.region=='sphere':
            from matplotlib.colors import LinearSegmentedColormap
            cmap=LinearSegmentedColormap.from_list('field_uniform_green',['#2766ae','#d6ebef','#54b77b','#f4cfbc','#c13740'])
        im=heat.pcolormesh(self.axis,self.axis,np.ma.masked_invalid(grid),cmap=cmap,vmin=-limit,vmax=limit,shading='auto')
        heat.set_aspect('equal');heat.set(xlabel='x / mm',ylabel='z / mm' if self.region=='sphere' else 'y / mm',
            title='DSV 子午截面相对场偏差 / ppm（固定色标）' if self.region=='sphere' else 'z=%g圆盘相对场偏差 / ppm（固定色标）'%self.z)
        values=norm[:len(self.map_points)];lo,hi=int(np.argmin(values)),int(np.argmax(values))
        plane_columns=[0,2] if self.region=='sphere' else [0,1]
        heat.scatter(*self.map_points[hi,plane_columns],marker='^',color='#111',s=28);heat.scatter(*self.map_points[lo,plane_columns],marker='v',color='#111',s=28)
        heat.text(.02,.02,'▲/▼ 仅预览网格极值，非全局极值证明',transform=heat.transAxes,fontsize=7)
        fig.colorbar(im,ax=heat,label='ppm',extend='both')
        line=axes[1,0];offset=len(self.map_points);base_mean=self.seed_metrics['mean_norm_T']
        base=(np.linalg.norm(self.base_fields[offset:],axis=1)-base_mean)/base_mean*1e6
        now=ppm[offset:]
        if self.region=='sphere':
            base=np.linalg.norm(self.base_fields[offset:],axis=1)*1000;now=norm[offset:]*1000
        for index,label in enumerate(('轴向 B(z)','径向 B(r)') if self.region=='sphere' else ('x直径线','y直径线')):
            sl=slice(index*61,(index+1)*61)
            line.plot(self.line_x,base[sl],color='#aaa',ls='--',lw=.8,label='起点 '+label)
            if self.previous_line is not None:line.plot(self.line_x,self.previous_line[sl],color='#6baed6',alpha=.25,lw=1)
            line.plot(self.line_x,now[sl],lw=1.4,label='当前 '+label)
        self.previous_line=now.copy();line.set(xlabel='平面内直径坐标 / mm',ylabel='相对各自面积均值 / ppm',title='两条平面直径线：不引入轴向/DSV评价')
        if self.region=='sphere':line.set(xlabel='轴向 / 径向坐标 / mm',ylabel='|B| / mT',title='轴向与赤道径向场分布')
        line.legend(fontsize=7,ncol=2);line.grid(alpha=.2)
        phase=axes[1,1];history=[e for e in self.events if e.get('metrics')]
        bx=[e['metrics']['mean_norm_T']*1e3 for e in history];py=[e['metrics']['pp_norm_ppm'] for e in history]
        if bx:
            phase.scatter(bx,py,s=8,c='#b7c1cd',alpha=.55)
            incumbent=None;path=[]
            for snapshot in history:
                if incumbent is None or feasibility_score(snapshot['metrics'],self.requirements)<feasibility_score(incumbent['metrics'],self.requirements):incumbent=snapshot
                path.append(incumbent['metrics'])
            phase.plot([m['mean_norm_T']*1e3 for m in path],[m['pp_norm_ppm'] for m in path],color='#df7a20',lw=.9,alpha=.8)
        population=event.get('population_metrics') or []
        if population:phase.scatter([m['mean_norm_T']*1e3 for m in population],[m['pp_norm_ppm'] for m in population],s=9,color='#2468ac',alpha=.5)
        phase.scatter([mean*1e3],[metrics['pp_norm_ppm']],s=48,c='#e8750b')
        bmin=self.requirements.get('min_mean_norm_T',0)*1000;pmax=self.requirements.get('max_pp_norm_ppm')
        phase.axvline(bmin,color='#2e8b57',ls='--',lw=.8)
        if pmax is not None:
            phase.axhline(pmax,color='#2e8b57',ls='--',lw=.8)
            xmax=max([bmin+1]+bx);phase.fill_between([bmin,xmax],[0,0],[pmax,pmax],color='#85c99d',alpha=.15)
        else:
            phase.axvspan(bmin,max([bmin+1]+bx),color='#85c99d',alpha=.12)
        phase.set(xlabel='训练面积平均场 / mT',ylabel='峰峰值 / ppm',title='最优快照轨迹；绿色为训练双门槛域')
        if pmax is None:phase.set_title('均衡推荐轨迹；绿色满足场强硬门槛')
        if self.region=='sphere':phase.set_xlabel('训练体积平均场 / mT')
        phase.set_yscale('symlog',linthresh=.01);phase.grid(alpha=.2)
        stage=event.get('stage_name',event.get('phase',event.get('event','')))
        count=event.get('global_evaluations',event.get('field_evaluations',0))
        fig.suptitle('MRI 演化 · %s · 搜索求场 %s · %.6f mT / %.4f ppm'%(stage,count,mean*1000,metrics['pp_norm_ppm']),fontsize=15)
        scope=('球形 DSV 直径%g mm；截面图仅为预览，极值仍按全域密集复核。'%self.diameter if self.region=='sphere'
            else '所有预览点固定 z=%g mm，直径%g mm。密集复核状态另列。'%(self.z,self.diameter))
        fig.text(.5,.012,'解析预览额外求场单列统计；'+scope,ha='center',fontsize=8,color='#53657b')
        fig.tight_layout(rect=(0,.035,1,.95))
        frame_path=self.folder/('frame_%04d.png'%self.render_count);fig.savefig(frame_path,dpi=100);plt.close(fig)
        # Replace a completed PNG atomically; an editor never reads a half image.
        preview=self.root/'实时演化看板.png';temporary=self.root/'实时演化看板.tmp.png'
        temporary.write_bytes(frame_path.read_bytes());os.replace(temporary,preview)
        frame={'image':frame_path.relative_to(self.root).as_posix(),'event_index':event['event_index'],
            'stage':stage,'evaluations':count,'mean_mT':mean*1000,'ppm':metrics['pp_norm_ppm']}
        if len(self.frames)>=self.max_frames:self.frames[-1]=frame
        else:self.frames.append(frame)
        self.render_count+=1

    def finish(self):
        if not self.enabled:return {'enabled':False}
        started=time.perf_counter();gif_file=None;timeline_file=None
        try:
            if self.frames:
                from PIL import Image
                pictures=[]
                for item in self.frames:
                    with Image.open(self.root/item['image']) as src:pictures.append(src.convert('RGB').resize((1200,675)))
                gif=self.root/'00_拓扑演化过程.gif';pictures[0].save(gif,save_all=True,append_images=pictures[1:],duration=int(self.options.get('gif_duration_ms',650)),loop=0,optimize=False)
                for pic in pictures:pic.close()
                gif_file=gif.name
                packed=json.dumps(self.frames,ensure_ascii=False).replace('</','<\\/')
                page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>MRI演化回放</title><style>body{font-family:Microsoft YaHei,sans-serif;margin:24px;color:#24344b}img{max-width:100%%;border:1px solid #d4deea}input{width:85%%}</style><h1>MRI 磁铁排布演化回放</h1><p>z=%g mm、直径%g mm圆盘；这是记录的迭代快照，不是物理时间瞬态。</p><input id="slider" type="range" min="0" max="%d" value="0"><button id="play">播放/暂停</button><p id="label"></p><img id="frame"><script>const frames=%s;let timer=null;const slider=document.getElementById('slider');function show(){const f=frames[+slider.value];document.getElementById('frame').src=f.image;document.getElementById('label').textContent=`${+slider.value+1}/${frames.length} · ${f.stage} · 求场${f.evaluations} · ${f.mean_mT.toFixed(6)}mT / ${f.ppm.toFixed(4)}ppm`;}slider.oninput=show;document.getElementById('play').onclick=()=>{if(timer){clearInterval(timer);timer=null;}else timer=setInterval(()=>{slider.value=(+slider.value+1)%%frames.length;show();},700);};show();</script></html>'''%(self.z,self.diameter,len(self.frames)-1,packed)
                target=self.root/'演化时间轴.html';target.write_text(page,encoding='utf-8');timeline_file=target.name
                if self.region=='sphere':
                    page=page.replace('z=%g mm、直径%g mm圆盘'%(self.z,self.diameter),'球形 DSV 直径%g mm'%self.diameter)
                    target.write_text(page,encoding='utf-8')
        except Exception as error:self.error=str(error)
        self.overhead_seconds+=time.perf_counter()-started
        result={'enabled':True,'preview_file':'实时演化看板.png' if self.frames else None,'gif_file':gif_file,
            'frames':[dict(frame) for frame in self.frames], 'frames_path_base':'run',
            'timeline_file':timeline_file,'history_file':self.history_path.relative_to(self.root).as_posix(),
            'frames_count':len(self.frames),'events_count':len(self.events),'render_count':self.render_count,
            'error':self.error,'preview_engine_statistics':dict(self.engine.statistics),
            'overhead_seconds':self.overhead_seconds,'preview_field_cache_hits':self.field_cache_hits,'max_render_frames':self.max_frames,'preview_cost_excluded_from_search_budget':True,
            'region':{'geometry':self.region,'diameter_mm':self.diameter,'z_mm':self.z},
            'snapshot_not_physical_time':True,'global_extrema_proven':False}
        save_json(self.folder/'evolution_manifest.json',result);return result
