"""Read-only COMSOL inspection and comparison with an existing saved solution.

This module uses only the Python standard library.  It never edits an MPH,
changes remanence, generates a mesh, or runs a study.  Interpolation of an
existing solution is deliberately separate from a new FEM verification.
"""

from __future__ import annotations

import csv
import hashlib
import io
import itertools
import json
import math
from pathlib import Path
import re
import subprocess
import time
from typing import Any, Mapping, Sequence
import uuid
import xml.etree.ElementTree as ET
import zipfile


_XML_LIMIT = 64 * 1024 * 1024
_COORD_FACTORS_MM = {"mm": 1.0, "cm": 10.0, "m": 1000.0}
_FIELD_FACTORS_T = {"T": 1.0, "mT": 1e-3, "uT": 1e-6, "µT": 1e-6}
_CSV_COLUMNS = ("x_mm", "y_mm", "z_mm", "Bx_T", "By_T", "Bz_T")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_member(archive: zipfile.ZipFile, name: str) -> bytes:
    info = archive.getinfo(name)
    if info.file_size > _XML_LIMIT:
        raise ValueError(f"Metadata member exceeds size limit: {name}")
    return archive.read(name)


def _xml(raw: bytes) -> ET.Element:
    if b"<!DOCTYPE" in raw.upper() or b"<!ENTITY" in raw.upper():
        raise ValueError("XML document type and entity declarations are unsupported")
    return ET.fromstring(raw)


def _typed_strings(raw: str) -> list[str]:
    """Extract COMSOL's serialized string cells without evaluating expressions."""
    return [s.replace("''", "'") for s in re.findall(r"'((?:[^']|'')*)'", raw)]


def _properties(node: ET.Element) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for child in node:
        if child.tag == "propertyValue":
            key = child.get("name", "").removeprefix("p:")
            value = child.get("value")
            if value is None and child.get("valueMatrix") is not None:
                value = _typed_strings(child.get("valueMatrix", ""))
            result[key] = value
            if child.get("Reference"):
                result[key + "_reference"] = child.get("Reference")
        elif child.tag == "param":
            values = _typed_strings(child.get("value", ""))
            result[child.get("param", "")] = values[0] if len(values) == 1 else values
    return result


def _selection(node: ET.Element) -> dict[str, Any]:
    selection = node.find("selection")
    if selection is None:
        return {}
    named = selection.findtext("named")
    explicit = selection.find("explicit")
    result: dict[str, Any] = {"type": selection.get("selType")}
    if named:
        result["named"] = named
    if explicit is not None:
        result["explicit_attributes"] = dict(explicit.attrib)
    return result


def _node_summary(node: ET.Element) -> dict[str, Any]:
    return {
        "tag": node.get("tag"), "label": node.get("name"),
        "operation": node.get("op"), "properties": _properties(node),
        "selection": _selection(node),
    }


def _document_summary(root: ET.Element) -> dict[str, Any]:
    parameters = {
        item.get("name"): item.get("expr")
        for group in root.iter("ModelParam") for item in group.findall("expressions")
    }
    variables = []
    for group in root.iter("Expr"):
        variables.append({
            "tag": group.get("tag"), "parent": group.findtext("storedParent"),
            "selection": _selection(group),
            "expressions": {e.get("name"): e.get("expr") for e in group.findall("expressions")},
        })
    physics = []
    for interface in root.iter("Physics"):
        physics.append({
            "tag": interface.get("tag"), "operation": interface.get("op"),
            "label": interface.get("name"),
            "features": [_node_summary(n) for n in interface.iter("PhysicsFeature")
                         if n.get("op") in ("MagneticFluxConservation", "MagneticInsulation")],
        })
    geometry_units = {e.get("tag"): e.findtext("lengthUnit") for e in root.iter("GeomSequence")}
    tables = []
    for table in root.iter("TableFeature"):
        rows = []
        for row in table.findall("realData"):
            try:
                rows.append([float(value) for value in (row.text or "").split(",")])
            except ValueError:
                continue
        tables.append({"tag": table.get("tag"), "label": table.get("name"),
                       "description": table.findtext("entityComments"),
                       "column_headers": _typed_strings(table.findtext("columnHeaders", "")),
                       "stored_real_rows": rows,
                       "warning": "Stored table values retain their original dataset and are not a DSV assessment."})
    return {
        "parameters": parameters, "variables": variables, "physics": physics,
        "geometry_length_units": geometry_units,
        "air_spheres": [_node_summary(n) for n in root.iter("GeomFeature") if n.get("op") == "Sphere"],
        "mesh_features": [_node_summary(n) for n in root.iter("MeshFeature")],
        "datasets": [_node_summary(n) for n in root.iter("DatasetFeature")],
        "numerical_evaluations": [_node_summary(n) for n in root.iter("NumericalFeature")],
        "studies": [{"tag": n.get("tag"), "label": n.get("name")} for n in root.iter("Study")],
        "solvers": [{"tag": n.get("tag"), "label": n.get("name")} for n in root.iter("SolverSequence")],
        "tables": tables,
    }


def inspect_mph(path: str | Path) -> dict[str, Any]:
    """Inspect ZIP/XML metadata, including the snapshot attached to a saved solve.

    Presence of solution binaries does not establish convergence or validate
    present-day design requirements.  COMSOL's MPH XML is an internal format;
    this reader reports observed properties and makes no compatibility promise.
    """
    source = Path(path).resolve(strict=True)
    before = sha256_file(source)
    with zipfile.ZipFile(source) as archive:
        names = archive.namelist()
        if "dmodel.xml" not in names:
            raise ValueError("MPH archive has no readable dmodel.xml")
        documents = {"current": _document_summary(_xml(_read_member(archive, "dmodel.xml")))}
        for name in names:
            if re.fullmatch(r"savepoint[^/]+/model\.zip", name):
                with zipfile.ZipFile(io.BytesIO(_read_member(archive, name))) as nested:
                    if "dmodel.xml" in nested.namelist():
                        documents[name.split("/")[0]] = _document_summary(_xml(_read_member(nested, "dmodel.xml")))
        info = dict(_xml(_read_member(archive, "modelinfo.xml")).attrib) if "modelinfo.xml" in names else {}
        version = _read_member(archive, "fileversion").decode("utf-8", "replace") if "fileversion" in names else ""
        solution_files = [n for n in names if re.fullmatch(r"solution.*\.mphbin", n)]
    after = sha256_file(source)
    if before != after:
        raise RuntimeError("Source MPH changed during read-only inspection")
    differences = {}
    for name, document in documents.items():
        if name == "current":
            continue
        differences[name] = [key for key in (
            "parameters", "variables", "physics", "geometry_length_units", "air_spheres", "mesh_features"
        ) if document[key] != documents["current"][key]]
    return {
        "source_path": str(source), "source_sha256": before,
        "source_size_bytes": source.stat().st_size, "source_unchanged": True,
        "file_version": version, "model_info": info,
        "has_saved_solution_binaries": bool(solution_files), "solution_binary_members": solution_files,
        "documents": documents, "current_vs_saved_snapshot_differences": differences,
        "validation_status": "metadata_only_not_mesh_or_FEM_convergence_validation",
    }


def _triples(values: Sequence[Sequence[float]], label: str) -> list[list[float]]:
    triples = []
    for index, row in enumerate(values):
        if len(row) != 3:
            raise ValueError(f"{label}[{index}] must have three components")
        converted = [float(value) for value in row]
        if not all(math.isfinite(value) for value in converted):
            raise ValueError(f"{label}[{index}] contains a non-finite value")
        triples.append(converted)
    if not triples:
        raise ValueError(f"{label} must not be empty")
    return triples


def write_field_csv(path: str | Path, points_mm: Sequence[Sequence[float]], fields_T: Sequence[Sequence[float]]) -> Path:
    points = _triples(points_mm, "points_mm")
    fields = _triples(fields_T, "fields_T")
    if len(points) != len(fields):
        raise ValueError("Point and field counts differ")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(_CSV_COLUMNS)
        writer.writerows([*point, *field] for point, field in zip(points, fields))
    return target


def read_field_csv(path: str | Path, coordinate_unit: str | None = None,
                   field_unit: str | None = None) -> dict[str, Any]:
    """Read explicit-unit x/y/z/Bx/By/Bz CSV; bare headers require both units.

    Accepted examples: x_mm,y_mm,z_mm,Bx_T,By_T,Bz_T and
    x (mm),y (mm),z (mm),mfnc.Bx (mT),mfnc.By (mT),mfnc.Bz (mT).
    No guesses from numerical magnitudes or from undocumented axis order.
    """
    text = Path(path).read_text(encoding="utf-8-sig")
    lines = [line.strip() for line in text.splitlines() if line.strip() and not line.lstrip().startswith("#")]
    # COMSOL commonly prefixes its header and metadata with a percent sign.
    lines = [line.lstrip("% ") if re.match(r"%\s*x(?:[_\s,(])", line) else line
             for line in lines if not line.startswith("%") or re.match(r"%\s*x(?:[_\s,(])", line)]
    if not lines:
        raise ValueError("CSV has no data")
    delimiter = "\t" if "\t" in lines[0] else ";" if ";" in lines[0] else ","
    rows = list(csv.reader(lines, delimiter=delimiter))
    columns = {}
    inferred_coords, inferred_fields = set(), set()
    for index, header in enumerate(rows[0]):
        normalized = header.strip().removeprefix("mfnc.")
        match = re.fullmatch(r"(Bx|By|Bz|x|y|z)(?:_([A-Za-zµ]+)|\s*[\[(]([A-Za-zµ]+)[\])])?", normalized)
        if not match:
            continue
        key, suffix1, suffix2 = match.groups()
        if key in columns:
            raise ValueError(f"Duplicate field column: {key}")
        columns[key] = index
        unit = suffix1 or suffix2
        if unit:
            (inferred_fields if key.startswith("B") else inferred_coords).add(unit)
    if set(columns) != {"x", "y", "z", "Bx", "By", "Bz"}:
        raise ValueError("CSV requires x,y,z,Bx,By,Bz columns with unambiguous units")
    if len(inferred_coords) > 1 or len(inferred_fields) > 1:
        raise ValueError("Mixed coordinate or field units are unsupported")
    header_coord = next(iter(inferred_coords), None)
    header_field = next(iter(inferred_fields), None)
    if coordinate_unit and header_coord and coordinate_unit != header_coord:
        raise ValueError("Coordinate unit conflicts with CSV header")
    if field_unit and header_field and field_unit != header_field:
        raise ValueError("Field unit conflicts with CSV header")
    coordinate_unit, field_unit = header_coord or coordinate_unit, header_field or field_unit
    if coordinate_unit not in _COORD_FACTORS_MM or field_unit not in _FIELD_FACTORS_T:
        raise ValueError("Explicit supported units required: coordinates mm/cm/m; fields T/mT/uT")
    points, fields = [], []
    for line_number, row in enumerate(rows[1:], 2):
        try:
            point = [float(row[columns[c]]) * _COORD_FACTORS_MM[coordinate_unit] for c in ("x", "y", "z")]
            field = [float(row[columns[c]]) * _FIELD_FACTORS_T[field_unit] for c in ("Bx", "By", "Bz")]
        except (ValueError, IndexError) as error:
            raise ValueError(f"Invalid CSV data row {line_number}") from error
        points.append(point)
        fields.append(field)
    return {"points_mm": _triples(points, "points_mm"), "fields_T": _triples(fields, "fields_T"),
            "source_path": str(Path(path).resolve()), "source_coordinate_unit": coordinate_unit,
            "source_field_unit": field_unit}


def compare_field_tables(predicted: Mapping[str, Any], reference: Mapping[str, Any],
                         coordinate_tolerance_mm: float = 1e-6) -> dict[str, Any]:
    """Compare one-to-one matched coordinates, preserving field sign and scale."""
    if not math.isfinite(coordinate_tolerance_mm) or coordinate_tolerance_mm <= 0:
        raise ValueError("coordinate_tolerance_mm must be finite and positive")
    pp = _triples(predicted["points_mm"], "predicted points")
    pf = _triples(predicted["fields_T"], "predicted fields")
    rp = _triples(reference["points_mm"], "reference points")
    rf = _triples(reference["fields_T"], "reference fields")
    if not (len(pp) == len(pf) == len(rp) == len(rf)):
        raise ValueError("Comparison requires equal non-empty point/field counts")
    tol = coordinate_tolerance_mm
    cells: dict[tuple[int, int, int], list[int]] = {}
    for i, point in enumerate(rp):
        cell = tuple(math.floor(v / tol) for v in point)
        cells.setdefault(cell, []).append(i)
    used, differences, maximum_coordinate_error = set(), [], 0.0
    for point, field in zip(pp, pf):
        cell = tuple(math.floor(v / tol) for v in point)
        matches = []
        for delta in itertools.product((-1, 0, 1), repeat=3):
            nearby = tuple(a + b for a, b in zip(cell, delta))
            for index in cells.get(nearby, []):
                distance = math.dist(point, rp[index])
                if distance <= tol:
                    matches.append((index, distance))
        if len(matches) != 1 or matches[0][0] in used:
            raise ValueError("Coordinates are missing, duplicated, or ambiguous within the requested tolerance")
        index, distance = matches[0]
        used.add(index)
        maximum_coordinate_error = max(maximum_coordinate_error, distance)
        differences.append([a - b for a, b in zip(field, rf[index])])
    count = len(differences)
    component = {}
    for axis, label in enumerate(("Bx", "By", "Bz")):
        errors = [row[axis] for row in differences]
        component[label] = {"bias_T": sum(errors) / count,
                            "rmse_T": math.sqrt(sum(x * x for x in errors) / count),
                            "maximum_absolute_error_T": max(abs(x) for x in errors)}
    norms = [math.sqrt(sum(v * v for v in row)) for row in differences]
    reference_mean_norm = sum(math.sqrt(sum(v * v for v in row)) for row in rf) / count
    vector_rmse = math.sqrt(sum(v * v for v in norms) / count)
    return {
        "matched_point_count": count, "coordinate_tolerance_mm": tol,
        "maximum_coordinate_error_mm": maximum_coordinate_error,
        "component_errors": component, "vector_rmse_T": vector_rmse,
        "maximum_vector_error_T": max(norms), "reference_mean_norm_T": reference_mean_norm,
        "vector_rmse_over_reference_mean_ppm": vector_rmse / reference_mean_norm * 1e6 if reference_mean_norm else None,
        "remanence_rescaled": False,
        "status": "pointwise_comparison_only_not_a_convergence_certificate",
    }


def prepare_existing_solution_export(mph_path: str | Path, points_mm: Sequence[Sequence[float]],
                                     output_dir: str | Path, comsol_root: str | Path,
                                     dataset_tag: str = "dset1", geometry_tag: str = "geom1",
                                     physics_tag: str = "mfnc") -> dict[str, Any]:
    """Create a Java/CLASS batch job that samples a saved solution, never solves."""
    source = Path(mph_path).resolve(strict=True)
    target = Path(output_dir).resolve()
    if target == source.parent:
        raise ValueError("Use a dedicated output subdirectory, not the folder holding the source MPH")
    for label in (dataset_tag, geometry_tag, physics_tag):
        if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", label):
            raise ValueError("COMSOL tags must be simple identifiers")
    points = _triples(points_mm, "points_mm")
    audit = inspect_mph(source)
    if not audit["has_saved_solution_binaries"]:
        raise ValueError("MPH contains no saved solution; this command will not solve it")
    current = audit["documents"]["current"]
    dataset = next((n for n in current["datasets"] if n["tag"] == dataset_tag), None)
    if not dataset or dataset["operation"] != "Solution":
        raise ValueError("An explicit saved Solution dataset is required")
    unit = current["geometry_length_units"].get(geometry_tag)
    if unit not in _COORD_FACTORS_MM:
        raise ValueError(f"Unsupported geometry length unit: {unit}")
    target.mkdir(parents=True, exist_ok=True)
    template = Path(__file__).resolve().parents[1] / "support" / "comsol" / "ExistingSolutionExport.java.template"
    source_code = template.read_text(encoding="utf-8")
    export_job_id = uuid.uuid4().hex
    initializers, calls = [], []
    for block_start in range(0, len(points), 200):
        block_index = block_start // 200
        lines = [f"  private static void fill{block_index}(double[][] points) {{"]
        for index in range(block_start, min(block_start + 200, len(points))):
            for axis in range(3):
                lines.append(f"    points[{axis}][{index}] = {points[index][axis]:.17g};")
        lines.append("  }")
        initializers.append("\n".join(lines))
        calls.append(f"    fill{block_index}(pointsMM);")
    replacements = {
        "@MPH_PATH@": json.dumps(source.as_posix(), ensure_ascii=True),
        "@DATASET@": json.dumps(dataset_tag), "@PHYSICS@": physics_tag,
        "@POINT_COUNT@": str(len(points)), "@COORD_SCALE@": str(1.0 / _COORD_FACTORS_MM[unit]),
        "@INITIALIZER_CALLS@": "\n".join(calls), "@INITIALIZER_METHODS@": "\n\n".join(initializers),
        "@JOB_ID@": export_job_id,
    }
    for token, replacement in replacements.items():
        source_code = source_code.replace(token, replacement)
    java_path = target / "MRIExistingSolutionExport.java"
    java_path.write_text(source_code, encoding="ascii")
    with (target / "requested_points_mm.csv").open("w", newline="", encoding="ascii") as stream:
        writer = csv.writer(stream); writer.writerow(("x_mm", "y_mm", "z_mm")); writer.writerows(points)
    root = Path(comsol_root).resolve()
    job = {
        "mode": "existing_solution_interpolation_only", "source_mph": str(source),
        "source_sha256": audit["source_sha256"], "geometry_coordinate_unit": unit,
        "point_count": len(points), "points_mm": points, "java_path": str(java_path),
        "output_dir": str(target), "output_csv": str(target / "existing_solution_field.csv"),
        "compile_command": [str(root / "bin" / "win64" / "comsolcompile.exe"), str(java_path)],
        "batch_command": [str(root / "bin" / "win64" / "comsolbatch.exe"), "-inputfile",
                          str(java_path.with_suffix(".class")), "-nosave", "-np", "2",
                          "-batchlog", str(target / "comsol_progress.log"), "-batchlogout"],
        "dataset_tag": dataset_tag, "physics_tag": physics_tag,
        "export_job_id": export_job_id,
        "comsol_log_dir": str(Path.home() / ".comsol" / "v62" / "logs"),
    }
    (target / "job.json").write_text(json.dumps(job, ensure_ascii=False, indent=2), encoding="utf-8")
    (target / "mph_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    return job


def run_existing_solution_export(job: Mapping[str, Any] | str | Path,
                                 timeout: float = 600) -> dict[str, Any]:
    """Compile and execute an audited generated job, checking source SHA256 twice."""
    if not isinstance(job, Mapping):
        job = json.loads(Path(job).read_text(encoding="utf-8"))
    target = Path(job["output_dir"])
    source = Path(job["source_mph"])
    expected = job["source_sha256"]
    if sha256_file(source) != expected:
        raise RuntimeError("Source MPH changed after job preparation")
    result = {"mode": job["mode"], "source_sha256_before": expected, "status": "running"}
    start = time.monotonic()
    try:
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        for phase in ("compile", "batch"):
            command = list(job[phase + "_command"])
            if not Path(command[0]).is_file():
                raise FileNotFoundError(command[0])
            logfile = target / (phase + ".log")
            with logfile.open("wb") as stream:
                completed = subprocess.run(command, cwd=target, stdout=stream,
                                           stderr=subprocess.STDOUT, timeout=timeout,
                                           creationflags=flags, check=False)
            result[phase + "_returncode"] = completed.returncode
            result[phase + "_log"] = str(logfile)
            if completed.returncode:
                raise RuntimeError(f"COMSOL {phase} failed; read {logfile}")
        rows = []
        log = (target / "batch.log").read_text(encoding="utf-8", errors="replace")
        progress_path = target / "comsol_progress.log"
        if progress_path.is_file():
            log += "\n" + progress_path.read_text(encoding="utf-8", errors="replace")
        # COMSOL redirects Java System.out/err into its own per-process log.
        # Read only logs bearing this generated task's unique marker.
        log_directory = Path(job.get("comsol_log_dir", ""))
        marker = "MRI_EXPORT_JOB_ID," + str(job.get("export_job_id", ""))
        if log_directory.is_dir() and job.get("export_job_id"):
            for log_path in log_directory.glob("batch*.log"):
                if log_path.stat().st_mtime < time.time() - (time.monotonic() - start) - 2:
                    continue
                internal_log = log_path.read_text(encoding="utf-8", errors="replace")
                if marker in internal_log:
                    log += "\n" + internal_log
                    result["comsol_internal_log"] = str(log_path)
        diagnostics = [line for line in log.splitlines() if (
            "MRI_EXPORT_JOB_ID," in line or "MRI_FIELD_" in line
            or "java.security.AccessControlException:" in line
            or "IllegalStateException:" in line or "Messages:" in line
        )]
        (target / "sampling_diagnostics.log").write_text("\n".join(dict.fromkeys(diagnostics)), encoding="utf-8")
        parsed_rows = set()
        for line in log.splitlines():
            if "MRI_FIELD_ROW," in line:
                payload = line.split("MRI_FIELD_ROW,", 1)[1]
                # -batchlogout can repeat a progress line in the captured stdout.
                if payload in parsed_rows:
                    continue
                parsed_rows.add(payload)
                values = payload.split(",")
                if len(values) != 6:
                    raise RuntimeError("Unexpected COMSOL sampling row shape")
                rows.append([float(v) for v in values])
        if len(rows) != job["point_count"]:
            if "java.security.AccessControlException:" in log:
                raise PermissionError("COMSOL's File system access preference blocks this Java read; "
                                      "no preferences were changed. See sampling_diagnostics.log")
            raise RuntimeError("COMSOL did not export every requested point; inspect batch.log")
        points = [row[:3] for row in rows]
        fields = [row[3:] for row in rows]
        # Match returned physical coordinates, not merely the order of a CSV.
        compare_field_tables({"points_mm": points, "fields_T": fields},
                             {"points_mm": job["points_mm"], "fields_T": fields})
        write_field_csv(job["output_csv"], points, fields)
        result["status"] = "success_existing_solution_only"
        result["output_csv"] = job["output_csv"]
        result["point_count"] = len(rows)
    except Exception as error:
        result["status"] = "failed"
        result["error"] = str(error)
        raise
    finally:
        result["elapsed_seconds"] = time.monotonic() - start
        result["source_sha256_after"] = sha256_file(source)
        result["source_unchanged"] = result["source_sha256_after"] == expected
        (target / "run_result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        if not result["source_unchanged"]:
            raise RuntimeError("Source MPH checksum changed; stop and inspect source provenance")
    return result
