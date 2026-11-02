"""Name only newly completed automatic runs; preserve path relocation evidence."""
from pathlib import Path
import json,copy,re
from urllib.parse import quote
from .objectives import feasibility_score
from .serialization import save_json


def _replace_paths(value,old,new,mapping):
    if isinstance(value,dict):
        result={}
        for key,item in value.items():
            if key.startswith('executed_') or key=='artifact_relocation':result[key]=item
            else:result[key]=_replace_paths(item,old,new,mapping)
        for key in ('compile_command','build_command'):
            if key in value and result[key]!=value[key]:result.setdefault('executed_'+key,copy.deepcopy(value[key]))
        return result
    if isinstance(value,list):return [_replace_paths(x,old,new,mapping) for x in value]
    if isinstance(value,str):
        for a,b in ((str(old),str(new)),(old.as_posix(),new.as_posix())):
            if value==a or value.startswith(a+'\\') or value.startswith(a+'/'):
                replaced=b+value[len(a):];mapping[value]=replaced;return replaced
    return value


def finish_run_name(public_dir,config,explicit_output=False):
    old=Path(public_dir).resolve();options=config.get('results',{})
    if not options.get('readable_names',False) or (explicit_output and not options.get('rename_explicit_output',False)):return old
    expected=(Path(config['parameters_file']).resolve().parent/'results').resolve()
    # Never move an arbitrary user folder, an input, or a historical run.
    if old.parent!=expected or not re.match(r'^\d{8}_\d{6}_\d{6}$',old.name):return old
    document=old/'详细数据/report_payload.json'
    if not document.is_file():return old
    payload=json.loads(document.read_text(encoding='utf-8'))
    candidates=[r for r in payload.get('candidates',[]) if r.get('feasible') and r.get('validation_metrics')]
    if not candidates:return old
    best=min(candidates,key=lambda r:feasibility_score(r['validation_metrics'],payload.get('requirements')))
    metrics=best['validation_metrics'];ppm=metrics['pp_norm_ppm']
    suffix='_最优%.2fmT_%sppm'%(metrics['mean_norm_T']*1000,format(ppm,'.3f') if ppm<10 else format(ppm,'.1f'))
    new=(old.parent/(old.name+suffix)).resolve()
    if new.parent!=expected or old==expected or new.exists():raise ValueError('Unsafe or occupied run rename target')
    old.rename(new)
    mapping={};changed=[]
    for file in new.rglob('*.json'):
        value=json.loads(file.read_text(encoding='utf-8-sig'));local={}
        replaced=_replace_paths(value,old,new,local)
        if local:
            if isinstance(replaced,dict):replaced['artifact_relocation']={'from':str(old),'to':str(new),'original_paths_recorded_in':'详细数据/run_relocation.json'}
            save_json(file,replaced);mapping.update(local);changed.append(file.relative_to(new).as_posix())
    save_json(new/'详细数据/run_relocation.json',{'from':str(old),'to':str(new),
        'paths':mapping,'updated_json_files':changed,'scope':'newly completed automatic run only; executed commands preserved separately'})
    link=quote(new.name)+'/'+quote('report_中文.html')
    (expected/'最新结果.html').write_text('<!doctype html><meta charset="utf-8"><meta http-equiv="refresh" content="0;url='+link+'"><a href="'+link+'">打开最新结果</a>',encoding='utf-8')
    return new
