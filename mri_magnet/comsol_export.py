"""Self-contained COMSOL Java/CLASS export of an actual candidate layout.

The generated class builds geometry, named magnet-domain selections, MFNC
remanence, and a stationary study.  It never loads a source MPH, meshes,
solves, or saves a model.  Compilation and optional model construction are
recorded separately from FEM verification.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import time
import uuid

from .geometry import envelope_limits, validate_design
from .identity import canonical_physical_sha256, raw_design_sha256, canonical_phase_deg, CANONICAL_SCHEMA_VERSION


_DEFAULTS = {
    "compile": True,
    "runtime_validate": False,
    "timeout_seconds": 300,
    "air_radius_mm": 650.0,
    "air_mu_r": 1.0,
    "mesh_size": 5,
    "class_prefix": "MRI_Design",
    "logs_dir": None,
    "native_processors": 2,
}


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _save(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def _settings(design, options):
    result = dict(_DEFAULTS)
    result.update(options or {})
    for key in ("air_radius_mm", "air_mu_r", "timeout_seconds"):
        result[key] = float(result[key])
        if not math.isfinite(result[key]) or result[key] <= 0:
            raise ValueError(key + " must be positive and finite")
    for key, lo, hi in (("mesh_size", 1, 9), ("native_processors", 1, 1024)):
        value = float(result[key])
        if not value.is_integer() or not lo <= value <= hi:
            raise ValueError("Invalid " + key)
        result[key] = int(value)
    if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,55}", str(result["class_prefix"])):
        raise ValueError("class_prefix must be a short ASCII Java identifier")
    limits = envelope_limits(design)
    checked = validate_design(design, limits)
    if not checked["valid"]:
        raise ValueError("Candidate geometry is invalid: " + "; ".join(checked["violations"]))
    if design.reference_mur <= 0 or not math.isfinite(design.reference_mur):
        raise ValueError("reference_mur must be positive and finite")
    needed = math.hypot(limits["outer_radius_mm"], limits["half_length_mm"])
    if result["air_radius_mm"] <= needed:
        raise ValueError("The air sphere must completely contain all magnet corners")
    return result


def _java_number(value):
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("Java numeric data must be finite")
    return format(value, ".17g")


def generate_java_model(design, settings=None):
    """Return class name, ASCII Java source and manifest from the same Design."""
    options = _settings(design, settings)
    physical = {key: value for key, value in design.to_dict().items() if key != "name"}
    design_hash = canonical_physical_sha256(design)
    raw_hash = raw_design_sha256(design)
    model_settings = {key: options[key] for key in ("air_radius_mm", "air_mu_r", "mesh_size")}
    model_hash = hashlib.sha256(json.dumps({"physical_sha256": design_hash, "physics": model_settings,
        "export_schema": "self-contained-v5-direction-grid-phase6"}, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    class_name = str(options["class_prefix"]) + "_" + model_hash[:12]
    direction_code = {"out": 0, "in": 1, "down": 2, "up": 3}
    ring_rows = []
    ring_ids = []
    ring_manifest = []
    direction_labels={"out":"N 径向朝外","in":"N 径向朝内","down":"N 轴向 -Z","up":"N 轴向 +Z"}
    direction_groups=[]
    ring_grid_ids={};grid_manifest=[]
    for polarity in direction_code:
        indices=[i for i,r in enumerate(design.rings) if r.polarity==polarity]
        if not indices:continue
        grids={}
        for i in indices:
            ring=design.rings[i]
            grids.setdefault((ring.count,round(canonical_phase_deg(ring.phase_deg,ring.count),6)),[]).append(i)
        grid_rows=[]
        for number,((count,canonical_phase),members) in enumerate(grids.items(),1):
            # Six decimals define grouping only. Keep the representative's
            # precise phase in the piecewise-constant Br direction expression.
            canonical_phase=canonical_phase_deg(design.rings[members[0]].phase_deg,count)
            suffix=polarity if len(grids)==1 else polarity+'_g%02d'%number
            grid={'grid_index':len(grid_manifest),'polarity':polarity,'count':count,
                  'phase_deg':design.rings[members[0]].phase_deg,
                  'canonical_phase_deg':canonical_phase,'ring_indices':members,
                  'ring_ids':[design.rings[i].id for i in members],
                  'magnet_count':sum(design.rings[i].count for i in members),
                  'domain_selection':'mag_'+suffix,'physics_feature':'pm_'+suffix,
                  'variable_collection':'dir_'+suffix,'angle_variable':'io_theta_'+suffix,
                  'direction_variables':['io_nx_'+suffix,'io_ny_'+suffix,'io_nz_'+suffix]}
            for i in members:ring_grid_ids[i]=grid['grid_index']
            grid_rows.append(grid);grid_manifest.append(grid)
        direction_groups.append({"polarity":polarity,"label":direction_labels[polarity],
            "domain_selection":"mag_"+polarity,"physics_feature":grid_rows[0]['physics_feature'] if len(grids)==1 else None,
            'physics_features':[grid['physics_feature'] for grid in grid_rows],
            "magnet_count":sum(design.rings[i].count for i in indices),
            "ring_ids":[design.rings[i].id for i in indices],"ring_indices":indices,
            "azimuth_grids":grid_rows})
    for index, ring in enumerate(design.rings):
        ring_rows.append("    {" + ", ".join(_java_number(v) for v in (ring.radius_mm, ring.z_mm, ring.count, ring.phase_deg, direction_code[ring.polarity])) + "}")
        ring_ids.append(json.dumps(ring.id, ensure_ascii=True))
        grid=grid_manifest[ring_grid_ids[index]]
        ring_manifest.append({**asdict(ring), "geometry_seed_tag": "b"+str(index), "geometry_rotate_tag": "r"+str(index),
                              "domain_selection": "mag_"+ring.polarity, "physics_feature":grid['physics_feature'],
                              'grid_selection':grid['domain_selection'],'grid_index':grid['grid_index'],
                              "geometry_domain_selections": ["geom1_b"+str(index)+"_dom","geom1_r"+str(index)+"_dom"]})
    rows = ",\n".join(ring_rows)
    ids = ", ".join(ring_ids)
    label = json.dumps("MRI 永磁阵列：" + str(design.name) + "（未求解）", ensure_ascii=True)
    source = '''import com.comsol.model.*;
import com.comsol.model.util.*;
import java.util.*;

/** Self-contained candidate model. Geometry and physics only; no mesh or study run. */
public class @CLASS@ {
  private static final String DESIGN_SHA = "@DESIGN_SHA@";
  private static final String RAW_DESIGN_SHA = "@RAW_DESIGN_SHA@";
  // radius_mm, z_mm, count, phase_deg, polarity(out=0,in=1,down=2,up=3)
  private static final double[][] RINGS = new double[][] {
@ROWS@
  };
  private static final String[] IDS = new String[] {@IDS@};
  private static final String[] POLARITIES = new String[] {"out","in","down","up"};
  private static final String[] DIRECTION_LABELS = new String[] {@DIRECTION_LABELS@};
  private static final int[] GRID_IDS = new int[] {@GRID_IDS@};
  private static final double[] CANONICAL_PHASES = new double[] {@CANONICAL_PHASES@};

  private static void addRing(Model model, GeomSequence g, int i) {
    double[] ring = RINGS[i];
    int count = (int)ring[2];
    double angle = ring[3] * Math.PI / 180.0;
    String block = "b" + i, rotation = "r" + i;
    g.create(block, "Block");
    g.feature(block).label(IDS[i] + " / seed");
    g.feature(block).set("base", "center");
    g.feature(block).set("size", new String[] {"io_cube", "io_cube", "io_cube"});
    g.feature(block).set("pos", new double[] {ring[0]*Math.cos(angle), ring[0]*Math.sin(angle), ring[1]});
    g.feature(block).set("rot", ring[3]);
    g.feature(block).set("selresult", "on");
    double[] angles = new double[count-1];
    for (int k=1; k<count; k++) angles[k-1] = k*360.0/count;
    g.create(rotation, "Rotate");
    g.feature(rotation).label(IDS[i] + " / circular copies");
    g.feature(rotation).selection("input").set(new String[] {block});
    g.feature(rotation).set("axistype", "z");
    g.feature(rotation).set("pos", new double[] {0,0,0});
    g.feature(rotation).set("rot", angles);
    g.feature(rotation).set("keep", "on");
    g.feature(rotation).set("selresult", "on");
  }

  private static int[] ringDomains(Model model, int i) {
    Set<Integer> unique = new TreeSet<Integer>();
    for (String tag : new String[] {"geom1_b"+i+"_dom", "geom1_r"+i+"_dom"}) {
      for (int domain : model.component("comp1").selection(tag).entities(3)) unique.add(domain);
    }
    int[] domains = new int[unique.size()]; int k=0;
    for (int domain : unique) domains[k++]=domain;
    if (domains.length != (int)RINGS[i][2]) {
      throw new IllegalStateException("Ring domain count mismatch: "+IDS[i]+" expected="+(int)RINGS[i][2]+" actual="+domains.length);
    }
    return domains;
  }

  private static String angleExpression(double phaseDegrees, int count) {
    double phase=phaseDegrees*Math.PI/180.0,step=2*Math.PI/count;
    return "("+phase+"+"+step+"*round((atan2(y,x)-("+phase+"))/"+step+"))";
  }

  private static void addUnion(Model model, String tag, String label, List<Integer> members, int[][] ringDomains) {
    List<String> inputs=new ArrayList<String>();Set<Integer> expected=new TreeSet<Integer>();
    for (int i : members) {
      inputs.add("geom1_b"+i+"_dom");inputs.add("geom1_r"+i+"_dom");
      for (int domain : ringDomains[i]) expected.add(domain);
    }
    model.component("comp1").selection().create(tag,"Union");
    model.component("comp1").selection(tag).label(label);
    model.component("comp1").selection(tag).set("entitydim",3);
    model.component("comp1").selection(tag).set("input",inputs.toArray(new String[0]));
    Set<Integer> actual=new TreeSet<Integer>();
    for (int domain : model.component("comp1").selection(tag).entities(3)) actual.add(domain);
    if (!actual.equals(expected)) throw new IllegalStateException("Grouped domain mismatch: "+tag);
  }

  private static void verifyGridDirection(Model model, String expression, String selection, List<Integer> members) {
    // Use the smallest radius as the worst angular cube extent. The angle
    // expression is constant on a cube only if every corner quantizes to its
    // column axis. Verify it with COMSOL's own expression evaluator, not only
    // a Java reimplementation. These are analytic parameter evaluations;
    // no mesh, solution, or result dataset is created.
    int index=members.get(0);
    for (int i : members) if (RINGS[i][0]<RINGS[index][0]) index=i;
    double[] ring=RINGS[index];double half=@CUBE@/2.0,maxError=0.;
    for (int column : new int[] {0,(int)ring[2]-1}) {
      double theta=(ring[3]+column*360.0/ring[2])*Math.PI/180.0;
      for (int dr : new int[] {-1,1}) for (int dt : new int[] {-1,1}) {
        double x=(ring[0]+dr*half)*Math.cos(theta)-dt*half*Math.sin(theta);
        double y=(ring[0]+dr*half)*Math.sin(theta)+dt*half*Math.cos(theta);
        String at="atan2(("+y+")[mm],("+x+")[mm])";
        model.param().set("io_verify_theta",expression.replace("atan2(y,x)",at));
        double value=model.param().evaluate("io_verify_theta");
        maxError=Math.max(maxError,Math.hypot(Math.cos(value)-Math.cos(theta),Math.sin(value)-Math.sin(theta)));
      }
    }
    model.param().remove("io_verify_theta");
    if (maxError>1e-9) throw new IllegalStateException("Grid direction varies across a cuboid: "+selection+" error="+maxError);
    String summary="MRI_GRID_DIRECTION_CHECK,"+DESIGN_SHA+","+selection+","+maxError;
    System.out.println(summary);ModelUtil.serverLog(summary);
  }

  private static void addGridPhysics(Model model, int polarity, String suffix, String selection, List<Integer> members) {
    String feature="pm_"+suffix,variables="dir_"+suffix,thetaVariable="io_theta_"+suffix;
    String[] direction=new String[] {"io_nx_"+suffix,"io_ny_"+suffix,"io_nz_"+suffix};
    double[] ring=RINGS[members.get(0)];
    model.component("comp1").variable().create(variables);
    model.component("comp1").variable(variables).label(DIRECTION_LABELS[polarity]+" / N="+(int)ring[2]+" / phase="+CANONICAL_PHASES[GRID_IDS[members.get(0)]]);
    model.component("comp1").variable(variables).selection().geom("geom1",3);
    model.component("comp1").variable(variables).selection().named(selection);
    String expression=angleExpression(ring[3],(int)ring[2]);
    model.component("comp1").variable(variables).set(thetaVariable,expression);
    String sign=polarity==0 ? "" : "-";
    model.component("comp1").variable(variables).set(direction[0],polarity<2 ? sign+"cos("+thetaVariable+")" : "0");
    model.component("comp1").variable(variables).set(direction[1],polarity<2 ? sign+"sin("+thetaVariable+")" : "0");
    model.component("comp1").variable(variables).set(direction[2],polarity<2 ? "0" : polarity==2 ? "-1" : "1");
    if (!model.component("comp1").variable(variables).selection().named().equals(selection)) throw new IllegalStateException("Grid variable selection mismatch");
    model.component("comp1").physics("mfnc").create(feature,"MagneticFluxConservation",3);
    model.component("comp1").physics("mfnc").feature(feature).label(DIRECTION_LABELS[polarity]+" / N="+(int)ring[2]+" / phase="+CANONICAL_PHASES[GRID_IDS[members.get(0)]]);
    model.component("comp1").physics("mfnc").feature(feature).selection().named(selection);
    model.component("comp1").physics("mfnc").feature(feature).set("ConstitutiveRelationBH","RemanentFluxDensity");
    model.component("comp1").physics("mfnc").feature(feature).set("mur_crel_BH_RemanentFluxDensity_mat","userdef");
    model.component("comp1").physics("mfnc").feature(feature).set("mur_crel_BH_RemanentFluxDensity",new String[] {"io_mur","0","0","0","io_mur","0","0","0","io_mur"});
    model.component("comp1").physics("mfnc").feature(feature).set("normBr_crel_BH_RemanentFluxDensity_mat","userdef");
    model.component("comp1").physics("mfnc").feature(feature).set("normBr_crel_BH_RemanentFluxDensity","io_Br");
    model.component("comp1").physics("mfnc").feature(feature).set("e_crel_BH_RemanentFluxDensity",direction);
    model.component("comp1").physics("mfnc").feature(feature).set("materialType","solid");
    if (polarity<2) verifyGridDirection(model,expression,selection,members);
    int domains=model.component("comp1").selection(selection).entities(3).length;
    String summary="MRI_GRID_GROUP,"+DESIGN_SHA+","+selection+","+feature+","+POLARITIES[polarity]+","+(int)ring[2]+","+CANONICAL_PHASES[GRID_IDS[members.get(0)]]+","+domains+","+members.size();
    System.out.println(summary);ModelUtil.serverLog(summary);
  }

  private static int addMagnetGroup(Model model, int polarity, int[][] ringDomains) {
    List<Integer> members=new ArrayList<Integer>();
    Map<Integer,List<Integer>> grids=new LinkedHashMap<Integer,List<Integer>>();
    for (int i=0;i<RINGS.length;i++) if ((int)RINGS[i][4]==polarity) {
      members.add(i);
      if (!grids.containsKey(GRID_IDS[i])) grids.put(GRID_IDS[i],new ArrayList<Integer>());
      grids.get(GRID_IDS[i]).add(i);
    }
    if (members.isEmpty()) return 0;
    String selection="mag_"+POLARITIES[polarity];
    addUnion(model,selection,DIRECTION_LABELS[polarity],members,ringDomains);
    int[] domains=model.component("comp1").selection(selection).entities(3);
    int number=0;
    for (List<Integer> gridMembers : grids.values()) {
      number++;
      String suffix=POLARITIES[polarity]+(grids.size()==1 ? "" : String.format(java.util.Locale.ROOT,"_g%02d",number));
      String gridSelection="mag_"+suffix;
      if (grids.size()>1) addUnion(model,gridSelection,DIRECTION_LABELS[polarity]+" / grid "+number,gridMembers,ringDomains);
      addGridPhysics(model,polarity,suffix,gridSelection,gridMembers);
    }
    String summary="MRI_DIRECTION_GROUP,"+DESIGN_SHA+","+POLARITIES[polarity]+","+domains.length+","+members.size();
    System.out.println(summary);ModelUtil.serverLog(summary);
    return domains.length;
  }

  public static Model run() {
    Model model = ModelUtil.createUnique("MRI");
    model.label(@LABEL@);
    model.param().set("io_cube","@CUBE@[mm]");
    model.param().set("io_Br","@BR@[T]");
    model.param().set("io_mur","@MUR@");
    model.param().set("air_mur","@AIR_MUR@");
    model.component().create("comp1",true);
    model.component("comp1").geom().create("geom1",3);
    GeomSequence g = model.component("comp1").geom("geom1");
    g.lengthUnit("mm");
    for (int i=0; i<RINGS.length; i++) addRing(model,g,i);
    g.create("air","Sphere");
    g.feature("air").label("Air sphere");
    g.feature("air").set("r","@AIR_RADIUS@[mm]");
    g.feature("air").set("pos",new double[] {0,0,0});
    // Form Union retains interior domains by default in COMSOL 6.2.
    g.feature("fin").set("action","union");
    g.run();
    model.component("comp1").physics().create("mfnc","MagnetostaticsNoCurrents","geom1");
    model.component("comp1").physics("mfnc").label("Permanent magnets / unsolved");
    model.component("comp1").physics("mfnc").feature("mfc1").set("mur_mat","userdef");
    model.component("comp1").physics("mfnc").feature("mfc1").set("mur",new String[] {"air_mur","0","0","0","air_mur","0","0","0","air_mur"});
    model.component("comp1").physics("mfnc").feature("mfc1").set("materialType","nonSolid");
    int selectedCount=0;
    int[][] ringDomainIds=new int[RINGS.length][];
    Set<Integer> uniqueDomains=new TreeSet<Integer>();
    for (int i=0;i<RINGS.length;i++) {
      ringDomainIds[i]=ringDomains(model,i);
      for (int domain : ringDomainIds[i]) {
        if (!uniqueDomains.add(domain)) throw new IllegalStateException("Magnet domain selected twice: "+domain);
      }
    }
    for (int polarity=0;polarity<4;polarity++) selectedCount+=addMagnetGroup(model,polarity,ringDomainIds);
    if (selectedCount != @COUNT@ || uniqueDomains.size() != @COUNT@) throw new IllegalStateException("Total magnet domain mismatch");
    if (g.getNDomains() != @COUNT@+1) throw new IllegalStateException("Expected magnets plus one air domain, actual="+g.getNDomains());
    model.component("comp1").mesh().create("mesh1");
    model.component("comp1").mesh("mesh1").autoMeshSize(@MESH@);
    model.study().create("std1");
    model.study("std1").create("stat","Stationary");
    model.study("std1").feature("stat").activate("mfnc",true);
    String summary = "MRI_CANDIDATE_BUILT,"+DESIGN_SHA+","+selectedCount+","+RINGS.length+","+g.getNDomains()+",@BR@,@MUR@";
    System.out.println(summary);
    ModelUtil.serverLog(summary);
    return model;
  }

  public static void main(String[] args) { run(); }
}
'''
    replacements = {"@GRID_IDS@":', '.join(str(ring_grid_ids[i]) for i in range(len(design.rings))),
                    '@CANONICAL_PHASES@':', '.join(_java_number(g['canonical_phase_deg']) for g in grid_manifest),
                    "@DIRECTION_LABELS@": ", ".join(json.dumps(direction_labels[p],ensure_ascii=True) for p in direction_code),
                    "@CLASS@": class_name, "@DESIGN_SHA@": design_hash, "@RAW_DESIGN_SHA@": raw_hash, "@ROWS@": rows, "@IDS@": ids,
                    "@LABEL@": label, "@CUBE@": _java_number(design.cube_mm), "@BR@": _java_number(design.remanence_T),
                    "@MUR@": _java_number(design.reference_mur), "@AIR_MUR@": _java_number(options["air_mu_r"]),
                    "@AIR_RADIUS@": _java_number(options["air_radius_mm"]), "@COUNT@": str(design.magnet_count), "@MESH@": str(options["mesh_size"])}
    for token, replacement in replacements.items():
        source = source.replace(token, replacement)
    if re.search(r"@[A-Z_]+@", source):
        raise RuntimeError("Unresolved Java template token")
    source.encode("ascii")
    manifest = {"schema_version": 5, "export_schema": "self-contained-v5-direction-grid-phase6", "class_name": class_name, "design_sha256": design_hash,
                "raw_design_sha256": raw_hash, "canonical_physical_sha256": design_hash,
                "canonical_schema_version": CANONICAL_SCHEMA_VERSION,
                "model_sha256": model_hash, "design_name": design.name, "magnet_count": design.magnet_count,
                "ring_count": len(design.rings), "cube_mm": design.cube_mm, "remanence_T": design.remanence_T,
                "reference_mur": design.reference_mur, "physics_settings": model_settings,
                "direction_counts": dict(Counter({p:sum(r.count for r in design.rings if r.polarity == p) for p in ("out","in","down","up")})),
                "rings": ring_manifest, "direction_groups": direction_groups,'grid_groups':grid_manifest,
                "selection_grouping": "polarity navigation unions with (polarity,count,canonical_phase) grid scopes",
                'coordinate_system_strategy':'COMSOL 6.2 supported selection-scoped variables; explicit global Br components; no unused local coordinate nodes',
                "direction_grid_membership": "grid ID assigned from canonical count/phase tuple before geometry construction; no domain-enumeration expressions",
                'direction_selection_count':len(direction_groups),'azimuth_grid_count':len(grid_manifest),
                "magnetic_selection_count": len(direction_groups)+sum(len(g['azimuth_grids']) if len(g['azimuth_grids'])>1 else 0 for g in direction_groups),
                "magnetic_physics_feature_count": len(grid_manifest),
                "magnetization_model": "piecewise-constant cuboid remanence; selection-scoped quantized azimuth variables per homogeneous grid; no domain branches",
                "geometry_units": "mm", "physics_interface": "MagnetostaticsNoCurrents / mfnc",
                "study": "Stationary, definition only", "mesh_built": False, "solve_executed": False,
                "native_model_constructed": False, "status": "java_generated_not_compiled"}
    return class_name, source, manifest


def _run(command, cwd, log_path, timeout):
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    with Path(log_path).open("wb") as stream:
        process = subprocess.Popen(command, cwd=cwd, stdout=stream, stderr=subprocess.STDOUT, creationflags=flags)
        try:
            return process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            if os.name == "nt":
                subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, creationflags=flags, check=False)
            else:
                process.kill()
            process.wait()
            raise


def _native_log_text(log_dir, started, marker):
    if not Path(log_dir).is_dir():
        return ""
    matching = []
    for path in Path(log_dir).glob("*.log"):
        try:
            if path.stat().st_mtime < started-5:
                continue
            with path.open("rb") as stream:
                stream.seek(max(0,path.stat().st_size-2_000_000))
                text = stream.read().decode("utf-8",errors="replace")
            matching.extend(line for line in text.splitlines() if marker in line)
        except OSError:
            continue
    return "\n".join(matching)


def export_comsol_model(design, output_dir, comsol_root, settings=None):
    """Generate a self-contained Java file, compile CLASS, optionally build only.

    ``compiled`` and ``model_built`` are separate.  A successful export never
    implies that an FEM mesh, solution or convergence check has been run.
    Failure is explicit in the returned status and a persistent manifest.
    """
    options = _settings(design, settings)
    name, source, manifest = generate_java_model(design, options)
    target = Path(output_dir).resolve(); target.mkdir(parents=True, exist_ok=True)
    logs = Path(options["logs_dir"]).resolve() if options["logs_dir"] else target
    logs.mkdir(parents=True, exist_ok=True)
    java_path = target/(name+".java"); class_path=java_path.with_suffix(".class")
    manifest_path=logs/(name+"_export.json"); compile_log=logs/(name+"_compile.log")
    java_path.write_text(source,encoding="ascii")
    result = {**manifest,"status":"java_generated_not_compiled","java_file":str(java_path),"class_file":None,
              "manifest_file":str(manifest_path),"compile_log":str(compile_log),"compiled":False,
              "model_built":False,"java_sha256":_sha256(java_path)}
    root=Path(comsol_root).resolve()
    started=time.time()
    try:
        if options["compile"]:
            compiler=root/"bin"/"win64"/"comsolcompile.exe"
            if not compiler.is_file():raise FileNotFoundError("COMSOL compiler not found: "+str(compiler))
            if class_path.exists():class_path.unlink()
            command=[str(compiler),str(java_path)]
            result["compile_command"]=command
            result["compile_returncode"]=_run(command,target,compile_log,options["timeout_seconds"])
            if result["compile_returncode"] or not class_path.is_file():
                raise RuntimeError("COMSOL compilation failed; inspect "+str(compile_log))
            if class_path.read_bytes()[:4] != b"\xca\xfe\xba\xbe":
                raise RuntimeError("Compiler output does not have a Java CLASS header")
            result.update(compiled=True,class_file=str(class_path),class_sha256=_sha256(class_path),status="compiled_class_ready_not_fem_verified")
        if options["runtime_validate"]:
            if not result["compiled"]:raise ValueError("runtime_validate requires a compiled class")
            batch=root/"bin"/"win64"/"comsolbatch.exe"
            if not batch.is_file():raise FileNotFoundError(str(batch))
            runtime_log=logs/(name+"_build.log"); progress_log=logs/(name+"_progress.log")
            command=[str(batch),"-inputfile",str(class_path),"-nosave","-np",str(options["native_processors"]),"-batchlog",str(progress_log),"-batchlogout"]
            result.update(build_command=command,build_log=str(runtime_log),build_progress_log=str(progress_log))
            result["build_returncode"]=_run(command,target,runtime_log,options["timeout_seconds"])
            marker="MRI_CANDIDATE_BUILT,"+result["design_sha256"]
            texts=[runtime_log.read_text(encoding="utf-8",errors="replace")]
            if progress_log.is_file():texts.append(progress_log.read_text(encoding="utf-8",errors="replace"))
            texts.append(_native_log_text(Path.home()/".comsol"/"v62"/"logs",started,marker))
            summary=next((line[line.index(marker):] for text in texts for line in text.splitlines() if marker in line),None)
            if result["build_returncode"] or not summary:
                raise RuntimeError("CLASS compiled, but native model build was not verified; inspect build logs")
            group_marker="MRI_DIRECTION_GROUP,"+result["design_sha256"]+","
            texts.append(_native_log_text(Path.home()/".comsol"/"v62"/"logs",started,group_marker))
            observed={}
            for text in texts:
                for line in text.splitlines():
                    if group_marker in line:
                        match=re.search(re.escape(group_marker)+r"(out|in|down|up),(\d+),(\d+)",line)
                        if match:observed[match.group(1)]={"magnet_count":int(match.group(2)),"ring_count":int(match.group(3))}
            expected={g['polarity']:{'magnet_count':g['magnet_count'],'ring_count':len(g['ring_indices'])} for g in result['direction_groups']}
            if observed!=expected:raise RuntimeError("Native direction groups do not match export manifest: "+str(observed))
            result['native_direction_groups']=observed
            grid_marker='MRI_GRID_GROUP,'+result['design_sha256']+','
            direction_marker='MRI_GRID_DIRECTION_CHECK,'+result['design_sha256']+','
            texts.append(_native_log_text(Path.home()/'.comsol'/'v62'/'logs',started,grid_marker))
            texts.append(_native_log_text(Path.home()/'.comsol'/'v62'/'logs',started,direction_marker))
            observed_grids={};direction_errors={}
            for text in texts:
                for line in text.splitlines():
                    if grid_marker in line:
                        match=re.search(re.escape(grid_marker)+r'([A-Za-z0-9_]+),([A-Za-z0-9_]+),(out|in|down|up),(\d+),([-+0-9.eE]+),(\d+),(\d+)',line)
                        if match:
                            selection,feature,polarity,count,phase,domains,members=match.groups()
                            observed_grids[selection]={'physics_feature':feature,'polarity':polarity,'count':int(count),
                                'canonical_phase_deg':float(phase),'magnet_count':int(domains),'ring_count':int(members)}
                    if direction_marker in line:
                        match=re.search(re.escape(direction_marker)+r'([A-Za-z0-9_]+),([-+0-9.eE]+)',line)
                        if match:direction_errors[match.group(1)]=float(match.group(2))
            expected_grids={g['domain_selection']:{'physics_feature':g['physics_feature'],'polarity':g['polarity'],
                'count':g['count'],'canonical_phase_deg':g['canonical_phase_deg'],'magnet_count':g['magnet_count'],
                'ring_count':len(g['ring_indices'])} for g in result['grid_groups']}
            if observed_grids!=expected_grids:raise RuntimeError('Native azimuth-grid groups do not match export manifest: '+str(observed_grids))
            expected_checks={g['domain_selection'] for g in result['grid_groups'] if g['polarity'] in ('out','in')}
            if set(direction_errors)!=expected_checks or any(not math.isfinite(v) or v>1e-9 for v in direction_errors.values()):
                raise RuntimeError('Native cuboid direction expression checks failed: '+str(direction_errors))
            result.update(native_grid_groups=observed_grids,native_direction_expression_errors=direction_errors,
                          native_direction_expressions_verified=True)
            values=summary.split(",")
            selected,rings,total=map(int,values[2:5])
            if selected!=design.magnet_count or rings!=len(design.rings) or total!=design.magnet_count+1:
                raise RuntimeError("Native geometry counters do not match the exported design")
            result.update(model_built=True,native_model_constructed=True,status="native_model_built_no_mesh_no_solution",
                          native_summary=summary,native_selected_magnet_domains=selected,native_ring_count=rings,native_domain_count=total)
    except Exception as error:
        result["status"]="native_build_failed" if result["compiled"] else "compile_failed"
        result["error"]=str(error)
    finally:
        result["elapsed_seconds"]=time.time()-started
        _save(manifest_path,result)
    return result
