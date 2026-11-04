"""Explicit new FEM solves and convergence checks, separate from saved-MPH reads.

Every Java task embeds geometry and observation coordinates. Only its own files
and child processes are written/controlled. Original Br and recoil permeability
are passed to COMSOL without any analytical demagnetization correction.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import time
import uuid
from dataclasses import replace

import numpy as np

from .comsol_export import generate_java_model, _native_log_text
from .comsol_bridge import write_field_csv,read_field_csv
from .geometry import Design
from .identity import canonical_physical_sha256,canonical_phase_deg
from .metrics import field_metrics
from .serialization import jsonable


DEFAULTS = {'mesh_levels': [7, 6, 5], 'air_radius_mm': 3000.,
    'air_expansion_factor': 1.5, 'magnet_hmax_mm': 4., 'target_hmax_mm': 4.,
    'refinement_factor': .7, 'native_processors': 2, 'timeout_seconds': 10800.,
    'mean_relative_tolerance': .005, 'ppm_absolute_tolerance': 10.,
    'ppm_relative_tolerance': .10, 'mesh_mean_relative_tolerance': .001,
    'convergence_ppm_fraction': 1./3., 'solver_relative_tolerance': 1e-10,
    'direct_solver_max_magnets': 1,
    'element_order':3,'max_air_expansions':6,
    'save_mph': False,'resume_completed':True,'symmetry_mode':'auto'}

# Acceptance ceilings are independent of product defaults. The product uses a
# tighter CG tolerance, while an older direct solve requested at 1e-8 can still
# be assessed honestly under the same fixed engineering contract.
ENGINEERING_LIMITS = {'mean_relative_tolerance':.005,'ppm_absolute_tolerance':10.,
    'ppm_relative_tolerance':.10,'mesh_mean_relative_tolerance':.001,
    'convergence_ppm_fraction':1./3.,'solver_relative_tolerance':1e-8}


def _save(path, value):
    Path(path).write_text(json.dumps(jsonable(value), ensure_ascii=False, indent=2,
                                   allow_nan=False), encoding='utf-8')


def _settings(settings=None):
    options = dict(DEFAULTS)
    if settings:
        unknown = set(settings)-set(options)
        if unknown: raise ValueError('Unknown FEM settings: '+str(sorted(unknown)))
        options.update(settings)
    levels = options['mesh_levels']
    if (not isinstance(levels, (list, tuple)) or len(levels)<3
            or any(isinstance(v,bool) or int(v)!=v or not 1<=v<=9 for v in levels)
            or any(a<=b for a,b in zip(levels,levels[1:]))):
        raise ValueError('mesh_levels must contain at least three successively finer COMSOL sizes')
    options['mesh_levels'] = list(map(int,levels))
    if options['symmetry_mode'] not in ('auto','full','rotational_sector'):
        raise ValueError('symmetry_mode must be auto, full or rotational_sector')
    for key in set(options)-{'mesh_levels','save_mph','resume_completed','symmetry_mode'}:
        value = float(options[key])
        if isinstance(options[key],bool) or not math.isfinite(value) or value<=0:
            raise ValueError(key+' must be positive and finite')
        options[key] = value
    # V6 engineering acceptance is a fixed public contract. Diagnostic callers
    # may request tighter limits, but a looser custom comparison must never
    # receive the same engineering certificate or green status.
    for key in ('mean_relative_tolerance','ppm_absolute_tolerance','ppm_relative_tolerance',
                'mesh_mean_relative_tolerance','convergence_ppm_fraction','solver_relative_tolerance'):
        if options[key]>ENGINEERING_LIMITS[key]:
            raise ValueError(key+' cannot exceed the V6 engineering acceptance limit')
    if not 0<options['refinement_factor']<1 or options['air_expansion_factor']<=1:
        raise ValueError('FEM refinement must decrease sizes and enlarge the air domain')
    if int(options['native_processors']) != options['native_processors']:
        raise ValueError('native_processors must be an integer')
    options['native_processors'] = int(options['native_processors'])
    if options['element_order'] not in (2,3):raise ValueError('element_order must be 2 or 3')
    if int(options['max_air_expansions'])!=options['max_air_expansions'] or options['max_air_expansions']>6:
        raise ValueError('max_air_expansions must be an integer between 1 and 6')
    if not isinstance(options['save_mph'],bool):raise ValueError('save_mph must be boolean')
    if not isinstance(options['resume_completed'],bool):raise ValueError('resume_completed must be boolean')
    return options


def _sample(sample):
    points=np.asarray(sample['points_mm'],float)
    if points.ndim!=2 or points.shape[1]!=3 or not len(points) or not np.all(np.isfinite(points)):
        raise ValueError('FEM observation points must be finite Nx3 mm')
    weights=np.asarray(sample.get('weights',np.ones(len(points))),float)
    if weights.shape!=(len(points),) or np.any(weights<0) or not np.all(np.isfinite(weights)) or weights.sum()<=0:
        raise ValueError('FEM sample requires matching nonnegative integration weights')
    return points,weights/weights.sum()


def sample_fingerprint(sample):
    """Frozen SI-independent mm coordinates and normalized volume weights."""
    points,weights=_sample(sample)
    digest=hashlib.sha256(b'v6-fem-sample-mm-f64-weights-v1')
    digest.update(np.asarray(points,dtype='<f8').tobytes(order='C'))
    # Re-normalizing a JSON round trip can change the last floating bit.
    # Eighteen decimals are far below any supported volume quadrature weight.
    digest.update(np.asarray(np.round(weights,18),dtype='<f8').tobytes(order='C'))
    return digest.hexdigest()


def rotational_sector(design,points):
    """One full angular period, bounded by exact mirror planes in air."""
    counts={ring.count for ring in design.rings}
    if len(counts)!=1:raise ValueError('FEM rotational sector requires identical column counts')
    count=int(next(iter(counts)));period=360./count
    phase=canonical_phase_deg(design.rings[0].phase_deg,count)
    for ring in design.rings:
        if abs(math.remainder(ring.phase_deg-phase,period))>1e-7:
            raise ValueError('FEM rotational sector requires compatible column phases and mirror planes')
        half=design.cube_mm/2
        if math.degrees(math.atan2(half,ring.radius_mm-half))>=period/2-1e-9:
            raise ValueError('A magnet intersects a sector mirror plane')
    start=math.radians(phase-period/2);width=2*math.pi/count
    points=np.asarray(points,float);theta=np.arctan2(points[:,1],points[:,0])
    mapped_theta=start+np.mod(theta-start,width)
    rho=np.hypot(points[:,0],points[:,1]);mapped=points.copy()
    mapped[:,0]=rho*np.cos(mapped_theta);mapped[:,1]=rho*np.sin(mapped_theta)
    metadata={'enabled':True,'rotation_order':count,'phase_deg':phase,
        'start_deg':math.degrees(start),'width_deg':period,'full_magnet_count':design.magnet_count,
        'sector_magnet_count':len(design.rings),
        'boundary_condition':'Exact meridian mirror planes: default Magnetic Insulation B.n=0',
        'equivalence_assumption':'identical uniform cuboids, common count/phase and radial or axial polarization; no tolerance perturbations'}
    return metadata,mapped,theta-mapped_theta


def restore_sector_fields(fields,angles):
    result=np.asarray(fields,float).copy();angle=np.asarray(angles,float)
    cs,sn=np.cos(angle),np.sin(angle);x=result[:,0].copy();y=result[:,1].copy()
    result[:,0]=cs*x-sn*y;result[:,1]=sn*x+cs*y
    return result


def _sector_geometry(source,design,air_radius,target_radius,sector,target_center_z=0.):
    # Keep full-array RINGS/Br metadata and direction expressions, but build
    # exactly the seed cube in each ring. No magnet is bisected by a seam.
    source=source.replace('double[] angles = new double[count-1];','double[] angles = new double[0];')
    start=source.index('    double[] angles = new double[0];')
    end=source.index('\n  }',start)
    source=source[:start]+source[end:]
    source=source.replace('new String[] {"geom1_b"+i+"_dom", "geom1_r"+i+"_dom"}',
                          'new String[] {"geom1_b"+i+"_dom"}')
    source=source.replace('inputs.add("geom1_b"+i+"_dom");inputs.add("geom1_r"+i+"_dom");',
                          'inputs.add("geom1_b"+i+"_dom");')
    source=source.replace('domains.length != (int)RINGS[i][2]','domains.length != 1')
    source=source.replace('selectedCount != %d || uniqueDomains.size() != %d'%(design.magnet_count,design.magnet_count),
        'selectedCount != %d || uniqueDomains.size() != %d'%(len(design.rings),len(design.rings)))
    source=source.replace('g.getNDomains() != %d+2'%design.magnet_count,'g.getNDomains() != %d+2'%len(design.rings))
    angle0=math.radians(sector['start_deg']);angle1=angle0+math.radians(sector['width_deg'])
    rows=[]
    for prefix,radius in (('air',air_radius),('dsv',target_radius)):
        wp=prefix+'Wedge';ext=prefix+'Prism';clip=prefix+'Clip';length=2*radius
        rows.extend([
            '    g.create("%s","WorkPlane");'%wp,
            '    g.feature("%s").set("quickplane","xy");'%wp,
            '    g.feature("%s").set("quickz",%.17g);'%(wp,(target_center_z if prefix=='dsv' else 0.)-radius),
            '    g.feature("%s").geom().create("p1","Polygon");'%wp,
            '    g.feature("%s").geom().feature("p1").set("x",new double[] {0,%.17g,%.17g});'%(wp,length*math.cos(angle0),length*math.cos(angle1)),
            '    g.feature("%s").geom().feature("p1").set("y",new double[] {0,%.17g,%.17g});'%(wp,length*math.sin(angle0),length*math.sin(angle1)),
            '    g.create("%s","Extrude");'%ext,
            '    g.feature("%s").selection("input").set(new String[] {"%s"});'%(ext,wp),
            '    g.feature("%s").set("distance",%.17g);'%(ext,2*radius),
            '    g.create("%s","Intersection");'%clip,
            '    g.feature("%s").selection("input").set(new String[] {"%s","%s"});'%(clip,prefix,ext),
            '    g.feature("%s").set("selresult","on");'%clip])
    source=source.replace('    // Form Union retains interior domains','\n'.join(rows)+'\n    // Form Union retains interior domains')
    source=source.replace('"geom1_dsv_dom"','"geom1_dsvClip_dom"')
    return source


def compare_fem_fields(predicted_fields, reference_fields, sample, settings=None):
    """Compare matching frozen points; FEM convergence is a separate requirement."""
    options=_settings(settings);points,weights=_sample(sample)
    predicted=np.asarray(predicted_fields,float);reference=np.asarray(reference_fields,float)
    if predicted.shape!=points.shape or reference.shape!=points.shape:
        raise ValueError('Python and FEM fields must match every requested point')
    pm=field_metrics(predicted,weights);fm=field_metrics(reference,weights)
    mean_error=abs(pm['mean_norm_T']/fm['mean_norm_T']-1.)
    ppm_error=abs(pm['pp_norm_ppm']-fm['pp_norm_ppm'])
    ppm_limit=max(options['ppm_absolute_tolerance'],options['ppm_relative_tolerance']*fm['pp_norm_ppm'])
    vector_error=np.linalg.norm(predicted-reference,axis=1)
    spectrum={'available':False,'reason':'Requires at least 81 globally distributed sphere-boundary points'}
    if sample.get('geometry')!='disk' and sample.get('n_surface') and not sample.get('symmetry_profile'):
        radius=float(sample['diameter_mm'])/2
        boundary=np.isclose(np.linalg.norm(points,axis=1),radius,rtol=1e-9,atol=1e-7)
        if np.count_nonzero(boundary)>=81:
            from .harmonics import fit_spherical_bz
            try:
                uniform=np.ones(np.count_nonzero(boundary))
                p=fit_spherical_bz(points[boundary],predicted[boundary],radius,uniform,lmax=8,reference_T=pm['mean_Bz_T'])
                f=fit_spherical_bz(points[boundary],reference[boundary],radius,uniform,lmax=8,reference_T=fm['mean_Bz_T'])
                spectrum={'available':True,'python':p,'fem':f,
                    'zonal_difference_ppm':{str(n):p['zonal_ppm'][str(n)]-f['zonal_ppm'][str(n)] for n in (2,4,6,8)}}
            except ValueError as error:spectrum={'available':False,'reason':str(error)}
    return {'python_metrics':pm,'fem_metrics':fm,'mean_relative_error':mean_error,
        'ppm_absolute_error':ppm_error,'mean_relative_limit':options['mean_relative_tolerance'],
        'ppm_error_limit':ppm_limit,'max_vector_error_T':float(vector_error.max()),
        'rms_vector_error_T':float(np.sqrt(weights@(vector_error*vector_error))),
        'component_bias_T':(weights@(predicted-reference)).tolist(),
        'harmonic_comparison':spectrum,
        'accuracy_passed':bool(mean_error<=options['mean_relative_tolerance'] and ppm_error<=ppm_limit),
        'scope':'same frozen samples and volume weights; convergence evaluated separately'}


def _convergence(first, second, sample, options):
    _,weights=_sample(sample)
    a=field_metrics(first,weights);b=field_metrics(second,weights)
    mean=abs(a['mean_norm_T']/b['mean_norm_T']-1.)
    ppm=abs(a['pp_norm_ppm']-b['pp_norm_ppm'])
    limit=max(options['ppm_absolute_tolerance'],options['ppm_relative_tolerance']*b['pp_norm_ppm'])*options['convergence_ppm_fraction']
    return {'mean_relative_change':mean,'ppm_change':ppm,'ppm_limit':limit,
        'mean_relative_limit':options['mesh_mean_relative_tolerance'],
        'passed':bool(mean<=options['mesh_mean_relative_tolerance'] and ppm<=limit)}


def _java_job(design, points, target_radius, task, options, task_id,sector=None):
    java_design=(replace(design,rings=tuple(replace(ring,phase_deg=canonical_phase_deg(ring.phase_deg,ring.count))
                 for ring in design.rings)) if sector else design)
    name,source,manifest=generate_java_model(java_design,{'air_radius_mm':task['air_radius_mm'],
        'mesh_size':task['mesh_size'],'class_prefix':'MRI_FEM'})
    # Different integration points and local mesh parameters need different classes.
    name2='MRI_FEM_'+hashlib.sha256((source+task_id).encode()).hexdigest()[:12]
    source=source.replace(name,name2)
    sphere='''    g.create("dsv", "Sphere");
    g.feature("dsv").set("r", "%s[mm]");
    g.feature("dsv").set("pos",new double[] {0,0,%s});
    g.feature("dsv").set("selresult", "on");
''' % (format(target_radius,'.17g'),format(task.get('target_center_z_mm',0.),'.17g'))
    source=source.replace('    // Form Union retains interior domains',sphere+'    // Form Union retains interior domains')
    # Freeze final-union repair in geometry units (mm); enlarging exterior air
    # must not enlarge a relative tolerance and close a real inter-magnet gap.
    source=source.replace('    g.run();',
        '    g.feature("fin").set("repairtoltype","absolute");\n'
        '    g.feature("fin").set("absrepairtol",1e-6);\n'
        '    g.run();')
    source=source.replace('g.getNDomains() != %d+1'%design.magnet_count,
                          'g.getNDomains() != %d+2'%design.magnet_count)
    source=source.replace('Expected magnets plus one air domain','Expected magnets plus target and exterior air domains')
    declarations=[];calls=[]
    for start in range(0,len(points),150):
        method='fillPoints%d'%(start//150);calls.append('    %s(points);'%method)
        body=['  private static void %s(double[][] p) {'%method]
        for index in range(start,min(start+150,len(points))):
            for axis in range(3):body.append('    p[%d][%d]=%s;'%(axis,index,format(points[index,axis],'.17g')))
        body.append('  }');declarations.append('\n'.join(body))
    groups=','.join(json.dumps(g['domain_selection']) for g in manifest['direction_groups'])
    tail='''
    model.component("comp1").physics("mfnc").prop("ShapeProperty").set("order_magneticscalarpotential",%d);
    model.component("comp1").physics("mfnc").create("v6gauge","ZeroMagneticScalarPotential",0);
    model.component("comp1").physics("mfnc").feature("v6gauge").selection().set(new int[] {1});
    model.component("comp1").selection().create("femMag","Union");
    model.component("comp1").selection("femMag").set("entitydim",3);
    model.component("comp1").selection("femMag").set("input",new String[] {%s});
    MeshSequence mesh=model.component("comp1").mesh("mesh1");
    mesh.create("magSize","Size");
    mesh.feature("magSize").selection().geom("geom1",3);
    mesh.feature("magSize").selection().named("femMag");
    mesh.feature("magSize").set("custom","on");
    mesh.feature("magSize").set("hmaxactive",true);
    mesh.feature("magSize").set("hmax","%s[mm]");
    mesh.create("dsvSize","Size");
    mesh.feature("dsvSize").selection().geom("geom1",3);
    mesh.feature("dsvSize").selection().named("geom1_dsv_dom");
    mesh.feature("dsvSize").set("custom","on");
    mesh.feature("dsvSize").set("hmaxactive",true);
    mesh.feature("dsvSize").set("hmax","%s[mm]");
    mesh.create("ftet1","FreeTet");
    String id="%s";
    System.out.println("MRI_FEM_STAGE,"+id+",mesh");
    ModelUtil.serverLog("MRI_FEM_STAGE,"+id+",mesh");
    mesh.run();
    System.out.println("MRI_FEM_STAGE,"+id+",solve");
    ModelUtil.serverLog("MRI_FEM_STAGE,"+id+",solve");
    model.study("std1").createAutoSequences("sol");
    model.sol("sol1").feature("s1").set("control","user");
    model.sol("sol1").feature("s1").set("stol",%s);
%s
    model.sol("sol1").runAll();
    double[][] points=new double[3][%d];
%s
    model.result().numerical().create("v6interp","Interp");
    model.result().numerical("v6interp").set("data","dset1");
    model.result().numerical("v6interp").set("expr",new String[] {"mfnc.Bx","mfnc.By","mfnc.Bz","x/1[mm]","y/1[mm]","z/1[mm]"});
    model.result().numerical("v6interp").set("unit",new String[] {"T","T","T","1","1","1"});
    model.result().numerical("v6interp").set("matherr","on");
    model.result().numerical("v6interp").set("ext",0.);
    model.result().numerical("v6interp").setInterpolationCoordinates(points);
    double[][][] values=model.result().numerical("v6interp").getData();
    if(values.length!=6||values[0].length!=1||values[0][0].length!=%d)throw new IllegalStateException("Unexpected FEM interpolation shape");
    for(int i=0;i<%d;i++) {
      for(int a=0;a<3;a++)if(!Double.isFinite(values[a][0][i])||Math.abs(values[a+3][0][i]-points[a][i])>1e-6)throw new IllegalStateException("Invalid FEM field or coordinate units");
      String row="MRI_FEM_ROW,"+id+","+i+","+values[0][0][i]+","+values[1][0][i]+","+values[2][0][i];
      System.out.println(row); ModelUtil.serverLog(row);
    }
    System.out.println("MRI_FEM_COMPLETE,"+id);
    ModelUtil.serverLog("MRI_FEM_COMPLETE,"+id);
'''%(int(options['element_order']),groups,format(task['magnet_hmax_mm'],'.17g'),format(task['target_hmax_mm'],'.17g'),task_id,
        format(options['solver_relative_tolerance'],'.17g'),
        ('    model.sol("sol1").feature("s1").create("v6direct","Direct");\n'
         '    model.sol("sol1").feature("s1").feature("v6direct").set("linsolver","pardiso");\n'
         '    model.sol("sol1").feature("s1").feature("fc1").set("linsolver","v6direct");'
         if (len(design.rings) if sector else design.magnet_count)<=options['direct_solver_max_magnets'] else
         '    model.sol("sol1").feature("s1").create("v6iter","Iterative");\n'
         '    model.sol("sol1").feature("s1").feature("v6iter").set("linsolver","cg");\n'
         '    model.sol("sol1").feature("s1").feature("v6iter").set("maxlinit",5000);\n'
         '    model.sol("sol1").feature("s1").feature("v6iter").create("v6amg","Multigrid");\n'
         '    model.sol("sol1").feature("s1").feature("v6iter").feature("v6amg").set("prefun","amg");\n'
         '    model.sol("sol1").feature("s1").feature("fc1").set("linsolver","v6iter");'),
        len(points),'\n'.join(calls),len(points),len(points))
    source=source.replace('    return model;\n  }\n\n  public static void main',tail+'    return model;\n  }\n\n  public static void main')
    source=source[:-2]+'\n'+'\n'.join(declarations)+'\n}\n'
    if sector:source=_sector_geometry(source,design,task['air_radius_mm'],target_radius,sector,task.get('target_center_z_mm',0.))
    return name2,source,manifest


def prepare_fem_validation(design, sample, output_dir, comsol_root, settings=None):
    """Generate four or more self-contained tasks; does not launch COMSOL."""
    if isinstance(design,dict):design=Design.from_dict(design)
    options=_settings(settings);points,weights=_sample(sample)
    sector=None;interpolation_points=points;rotation_angles=None
    requested_symmetry=options['symmetry_mode'];symmetry_reason=None
    if requested_symmetry=='auto':
        try:
            sector,interpolation_points,rotation_angles=rotational_sector(design,points)
            options['symmetry_mode']='rotational_sector'
        except ValueError as error:
            options['symmetry_mode']='full';symmetry_reason=str(error)
    if options['symmetry_mode']=='rotational_sector':
        sector,interpolation_points,rotation_angles=rotational_sector(design,points)
    center_z=float(sample.get('z_mm',0.)) if sample.get('geometry')=='disk' else 0.
    center=np.array([0.,0.,center_z])
    radius=max(1.,float(np.linalg.norm(points-center,axis=1).max())*1.02)
    # A meshing-only target sphere may never cut any magnetic cuboid.
    closest=min(math.hypot(max(0.,r.radius_mm-design.cube_mm/2),max(0.,abs(r.z_mm-center_z)-design.cube_mm/2)) for r in design.rings)
    if radius>=closest:raise ValueError('FEM target sphere intersects magnetic material')
    target=Path(output_dir).resolve();target.mkdir(parents=True,exist_ok=True)
    if (target/'fem_job.json').exists():raise FileExistsError('FEM verification directory already contains a job')
    jobid=uuid.uuid4().hex;tasks=[];root=Path(comsol_root).resolve()
    for index,level in enumerate(options['mesh_levels']):
        tasks.append({'name':'mesh_%02d'%(index+1),'mesh_size':level,'air_radius_mm':options['air_radius_mm'],
            'magnet_hmax_mm':options['magnet_hmax_mm']*options['refinement_factor']**index,
            'target_hmax_mm':options['target_hmax_mm']*options['refinement_factor']**index})
    tasks.append(dict(tasks[-1],name='expanded_air',air_radius_mm=options['air_radius_mm']*options['air_expansion_factor']))
    for index,task in enumerate(tasks):
        task['target_center_z_mm']=center_z
        taskid=jobid+'_%d'%index;folder=target/task['name'];folder.mkdir()
        name,source,manifest=_java_job(design,interpolation_points,radius,task,options,taskid,sector)
        java=folder/(name+'.java');java.write_text(source,encoding='ascii')
        task.update(id=taskid,folder=str(folder),java_file=str(java),class_file=str(java.with_suffix('.class')),
            java_sha256=hashlib.sha256(java.read_bytes()).hexdigest(),
            output_csv=str(folder/'field.csv'),manifest=manifest)
    job={'schema_version':6,'mode':'new_fem_mesh_solve_and_convergence','job_id':jobid,
        'output_dir':str(target),'comsol_root':str(root),'design':design.to_dict(),
        'design_sha256':canonical_physical_sha256(design),'sample_fingerprint':sample_fingerprint(sample),
        'sample':dict(sample,points_mm=points.tolist(),weights=weights.tolist()),
        'settings':options,'requested_symmetry_mode':requested_symmetry,
        'symmetry_resolution_reason':symmetry_reason,
        'target_mesh_radius_mm':radius,'tasks':tasks,'status':'prepared_not_solved',
        'symmetry':sector or {'enabled':False},'rotation_angles_rad':None if rotation_angles is None else rotation_angles.tolist(),
        'material_input':'original Br and recoil mu_r; no pre-corrected polarization'}
    _save(target/'fem_job.json',job);return job


def _execute(command, folder, log, timeout, cancel_check=None):
    flags=getattr(subprocess,'CREATE_NO_WINDOW',0)
    with Path(log).open('wb') as stream:
        process=subprocess.Popen(command,cwd=folder,stdout=stream,stderr=subprocess.STDOUT,creationflags=flags)
        started=time.monotonic()
        try:
            while process.poll() is None:
                if cancel_check:cancel_check()
                if time.monotonic()-started>timeout:raise TimeoutError('COMSOL task exceeded configured timeout')
                time.sleep(.5)
            return process.returncode
        except BaseException:
            if os.name=='nt':subprocess.run(['taskkill','/PID',str(process.pid),'/T','/F'],creationflags=flags,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
            else:process.kill()
            process.wait();raise


def parse_fem_rows(text, task_id, point_count):
    marker='MRI_FEM_ROW,'+task_id+',';rows={}
    for line in text.splitlines():
        if marker not in line:continue
        payload=line.split(marker,1)[1].strip()
        # serverLog appends its logger name after the numerical row; stdout
        # does not. Parse the same numerical payload on both paths.
        payload=re.sub(r'\s+\[[^\]\r\n]*\]\s*$','',payload)
        values=payload.split(',')
        if len(values)!=4:raise ValueError('Unexpected FEM row shape')
        index=int(values[0]);field=np.asarray(list(map(float,values[1:])))
        if not 0<=index<point_count or not np.all(np.isfinite(field)):raise ValueError('Invalid FEM row')
        if index in rows and not np.array_equal(rows[index],field):raise ValueError('Conflicting repeated FEM row')
        rows[index]=field
    if len(rows)!=point_count or 'MRI_FEM_COMPLETE,'+task_id not in text:
        raise RuntimeError('COMSOL did not complete and sample every requested point')
    return np.asarray([rows[i] for i in range(point_count)])


def _append_air_refinement(job,options):
    tasks=job['tasks'];last=tasks[-1];number=len(tasks)-len(options['mesh_levels'])+1
    folder=Path(job['output_dir'])/('expanded_air_%02d'%number);folder.mkdir()
    task={key:last[key] for key in ('mesh_size','magnet_hmax_mm','target_hmax_mm')}
    task.update(name=folder.name,air_radius_mm=last['air_radius_mm']*options['air_expansion_factor'],
                target_center_z_mm=last.get('target_center_z_mm',0.))
    points,_=_sample(job['sample']);design=Design.from_dict(job['design']);sector=None
    if job.get('symmetry',{}).get('enabled'):sector,points,_=rotational_sector(design,points)
    taskid=job['job_id']+'_'+str(len(tasks))
    name,source,manifest=_java_job(design,points,job['target_mesh_radius_mm'],task,options,taskid,sector)
    java=folder/(name+'.java');java.write_text(source,encoding='ascii')
    task.update(id=taskid,folder=str(folder),java_file=str(java),class_file=str(java.with_suffix('.class')),
        java_sha256=hashlib.sha256(java.read_bytes()).hexdigest(),output_csv=str(folder/'field.csv'),manifest=manifest)
    tasks.append(task);_save(Path(job['output_dir'])/'fem_job.json',job)


def _task_fingerprint(task,job):
    """Bind completed fields to the exact numerical model that produced them."""
    options=_settings(job['settings'])
    data={'version':'v6-fem-task-1','java_sha256':task['java_sha256'],
        'design_sha256':job['design_sha256'],'sample_fingerprint':job['sample_fingerprint'],
        'symmetry':job.get('symmetry',{'enabled':False}),
        'rotation_angles_rad':job.get('rotation_angles_rad'),
        'mesh':{key:task.get(key) for key in ('mesh_size','air_radius_mm','magnet_hmax_mm',
                                             'target_hmax_mm','target_center_z_mm')},
        'solver':{key:options[key] for key in ('element_order','solver_relative_tolerance',
                                              'direct_solver_max_magnets')}}
    return hashlib.sha256(json.dumps(jsonable(data),sort_keys=True,separators=(',',':'),
                                     allow_nan=False).encode('utf-8')).hexdigest()


def _cached_task(task,prior,points,job):
    candidate=next((item for item in prior.get('tasks',[]) if item.get('name')==task['name']),None)
    path=Path(task['output_csv'])
    if not candidate or candidate.get('status')!='solved_sampled' or not path.is_file():return None
    # Old reports without a numerical-model binding remain readable evidence,
    # but cannot be reused as an authenticated completed task after a restart.
    if candidate.get('task_fingerprint')!=_task_fingerprint(task,job):return None
    if hashlib.sha256(path.read_bytes()).hexdigest()!=candidate.get('field_sha256'):
        raise ValueError('Cached FEM field CSV changed')
    data=read_field_csv(path)
    if not np.array_equal(np.asarray(data['points_mm']),points):raise ValueError('Cached FEM coordinates changed')
    return dict(candidate,reused_completed_task=True),np.asarray(data['fields_T'])


def classify_fem_failure(message):
    """Classify actual COMSOL diagnostics without converting failure to proof."""
    text=str(message).lower()
    if any(token in text for token in ('outofmemory','out of memory','not enough memory','内存不足','java heap space')):
        return 'resource_memory_exhausted'
    if any(token in text for token in ('license checkout','license error','许可证','许可错误')):
        return 'license_unavailable'
    if any(token in text for token in ('identity changed','mapping changed','source changed','fingerprint')):
        return 'input_integrity_error'
    return 'solver_or_runtime_error'


def run_fem_validation(job, predicted_fields=None, event_callback=None, cancel_check=None):
    """Run bounded child tasks; a compiled CLASS is never reported as FEM verified."""
    if isinstance(job,(str,Path)):job=json.loads(Path(job).read_text(encoding='utf-8'))
    target=Path(job['output_dir']);options=_settings(job['settings']);sample=job['sample'];points,weights=_sample(sample)
    report={'status':'running','fem_verified':False,'model_accuracy_passed':False,
        'mesh_converged':False,'boundary_converged':False,'job_id':job['job_id'],
        'tasks':[],'sampled_extrema_only':True,'settings':options,
        'design_sha256':canonical_physical_sha256(job['design']),
        'sample_fingerprint':sample_fingerprint(sample)}
    report['symmetry']=job.get('symmetry',{'enabled':False})
    report['requested_symmetry_mode']=job.get('requested_symmetry_mode',options['symmetry_mode'])
    report['symmetry_resolution_reason']=job.get('symmetry_resolution_reason')
    actual_magnets=(len(job['design']['rings']) if report['symmetry'].get('enabled')
                    else sum(int(ring['count']) for ring in job['design']['rings']))
    report['linear_solver']='pardiso' if actual_magnets<=options['direct_solver_max_magnets'] else 'cg_amg'
    prior={};report_path=target/'fem_report.json'
    if options['resume_completed'] and report_path.is_file():
        prior=json.loads(report_path.read_text(encoding='utf-8'))
        if prior.get('design_sha256')!=report['design_sha256'] or prior.get('sample_fingerprint')!=report['sample_fingerprint']:
            prior={}
    started=time.monotonic();fields=[];root=Path(job['comsol_root'])/'bin'/'win64'
    try:
        if job.get('design_sha256')!=report['design_sha256'] or job.get('sample_fingerprint')!=report['sample_fingerprint']:
            raise ValueError('FEM job design or sampling identity changed after preparation')
        if job.get('symmetry',{}).get('enabled'):
            expected,_,angles=rotational_sector(Design.from_dict(job['design']),points)
            if job['symmetry']!=expected or not np.array_equal(np.asarray(job.get('rotation_angles_rad')),angles):
                raise ValueError('FEM rotational sample mapping changed after preparation')
        for task in job['tasks']:
            if cancel_check:cancel_check()
            folder=Path(task['folder']);java=Path(task['java_file'])
            if hashlib.sha256(java.read_bytes()).hexdigest()!=task['java_sha256']:raise ValueError('Generated FEM source changed after preparation')
            cached=_cached_task(task,prior,points,job)
            if cached:
                item,values=cached;report['tasks'].append(item)
            else:
                item={'name':task['name'],'status':'running','mesh_size':task['mesh_size'],
                      'air_radius_mm':task['air_radius_mm'],'magnet_hmax_mm':task['magnet_hmax_mm'],
                      'target_hmax_mm':task['target_hmax_mm'],
                      'java_sha256':task['java_sha256'],'task_fingerprint':_task_fingerprint(task,job)};report['tasks'].append(item)
                if event_callback:event_callback(dict(event='fem_task',**item))
                _save(target/'fem_report.json',report)
                started_task=time.time()
                for phase in ('compile','solve'):
                    command=([str(root/'comsolcompile.exe'),str(java)] if phase=='compile' else
                        [str(root/'comsolbatch.exe'),'-inputfile',task['class_file'],'-np',str(options['native_processors']),
                         '-batchlog',str(folder/'progress.log'),'-batchlogout'])
                    if phase=='solve':
                        command+=['-outputfile',str(folder/'solved.mph')] if options['save_mph'] else ['-nosave']
                    if not Path(command[0]).is_file():raise FileNotFoundError(command[0])
                    item[phase+'_command']=command;item['phase']=phase
                    _save(target/'fem_report.json',report)
                    rc=_execute(command,folder,folder/(phase+'.log'),options['timeout_seconds'],cancel_check)
                    item[phase+'_returncode']=rc
                    if rc:raise RuntimeError('COMSOL '+phase+' returned '+str(rc)+'; see '+str(folder/(phase+'.log')))
                text='\n'.join(p.read_text(encoding='utf-8',errors='replace') for p in (folder/'solve.log',folder/'progress.log') if p.exists())
                text+='\n'+_native_log_text(Path.home()/'.comsol'/'v62'/'logs',started_task,task['id'])
                (folder/'sampling_diagnostics.log').write_text(text,encoding='utf-8')
                values=parse_fem_rows(text,task['id'],len(points))
                if job.get('rotation_angles_rad') is not None:values=restore_sector_fields(values,job['rotation_angles_rad'])
                write_field_csv(task['output_csv'],points,values)
                item.update(status='solved_sampled',elapsed_seconds=time.time()-started_task,
                    metrics=field_metrics(values,weights),field_csv=task['output_csv'],
                    field_sha256=hashlib.sha256(Path(task['output_csv']).read_bytes()).hexdigest())
            fields.append(values)
            mesh_count=len(options['mesh_levels'])
            if len(fields)==len(job['tasks']) and len(fields)>mesh_count:
                boundary=_convergence(fields[-2],fields[-1],sample,options)
                if not boundary['passed'] and len(job['tasks'])-mesh_count<int(options['max_air_expansions']):
                    _append_air_refinement(job,options)
            _save(target/'fem_report.json',report)
        mesh=_convergence(fields[len(options['mesh_levels'])-2],fields[len(options['mesh_levels'])-1],sample,options)
        boundary=_convergence(fields[-2],fields[-1],sample,options)
        report.update(mesh_convergence=mesh,boundary_convergence=boundary,
            mesh_converged=mesh['passed'],boundary_converged=boundary['passed'],
            fem_verified=bool(mesh['passed'] and boundary['passed']),reference_field_csv=job['tasks'][-1]['output_csv'])
        report['reference_field_sha256']=report['tasks'][-1]['field_sha256']
        if predicted_fields is not None:
            comparison=compare_fem_fields(predicted_fields,fields[-1],sample,options)
            report.update(comparison=comparison,model_accuracy_passed=bool(report['fem_verified'] and comparison['accuracy_passed']))
        report['status']=('passed' if report['model_accuracy_passed'] else
            'fem_converged_model_accuracy_not_passed' if report['fem_verified'] else 'fem_not_converged')
        if predicted_fields is None and report['fem_verified']:report['status']='fem_converged_awaiting_python_comparison'
    except Exception as error:
        diagnostics=str(error)
        if report['tasks']:
            failed_name=report['tasks'][-1]['name']
            failed_task=next((item for item in job['tasks'] if item['name']==failed_name),None)
            if failed_task:
                for name in ('compile.log','solve.log','progress.log'):
                    path=Path(failed_task['folder'])/name
                    if path.exists():diagnostics+='\n'+path.read_text(encoding='utf-8',errors='replace')[-65536:]
        category=classify_fem_failure(diagnostics)
        report.update(status='resource_limited' if category=='resource_memory_exhausted' else 'failed',
                      failure_category=category,error=str(error),error_type=type(error).__name__)
        if report['tasks']:report['tasks'][-1]['status']='failed'
    except BaseException as error:
        report.update(status='cancelled',error=str(error),error_type=type(error).__name__)
        if report['tasks']:report['tasks'][-1]['status']='cancelled'
        raise
    finally:
        report['elapsed_seconds']=time.monotonic()-started
        _save(target/'fem_report.json',report)
    return report
