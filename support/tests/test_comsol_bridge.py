import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from mri_magnet.comsol_bridge import (
    compare_field_tables, inspect_mph, prepare_existing_solution_export,
    read_field_csv, write_field_csv, run_existing_solution_export, sha256_file,
)


class ComsolBridgeTests(unittest.TestCase):
    def test_unit_conversion_and_coordinate_matching_preserve_sign(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "reference.csv"
            path.write_text("x_m,y_m,z_m,Bx_mT,By_mT,Bz_mT\n0.01,0,0,0,0,-40\n0,0,0,0,0,-39\n")
            reference = read_field_csv(path)
            predicted = {"points_mm": [[0, 0, 0], [10, 0, 0]],
                         "fields_T": [[0, 0, -0.039], [0, 0, -0.04]]}
            comparison = compare_field_tables(predicted, reference)
            self.assertEqual(comparison["vector_rmse_T"], 0)
            self.assertFalse(comparison["remanence_rescaled"])
            wrong_sign = dict(predicted, fields_T=[[0, 0, .039], [0, 0, .04]])
            self.assertGreater(compare_field_tables(wrong_sign, reference)["vector_rmse_T"], .07)

    def test_unknown_units_and_ambiguous_coordinates_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "unknown.csv"
            path.write_text("x,y,z,Bx,By,Bz\n0,0,0,0,0,40\n")
            with self.assertRaises(ValueError):
                read_field_csv(path)
            self.assertEqual(read_field_csv(path, "mm", "mT")["fields_T"][0][2], .04)
        duplicate = {"points_mm": [[0, 0, 0], [0, 0, 0]], "fields_T": [[0, 0, .04], [0, 0, .04]]}
        with self.assertRaises(ValueError):
            compare_field_tables(duplicate, duplicate)

    def test_round_trip_explicit_units(self):
        with tempfile.TemporaryDirectory() as directory:
            path = write_field_csv(Path(directory) / "data.csv", [[10, 2, -3]], [[.001, 0, -.04]])
            self.assertEqual(read_field_csv(path)["points_mm"], [[10, 2, -3]])
            with self.assertRaises(ValueError):
                read_field_csv(path, coordinate_unit="m")

    def test_mph_current_and_saved_snapshot_are_reported_without_editing(self):
        xml = '''<Model><ModelParam><expressions name="io_Br" expr="1.43[T]"/></ModelParam>
        <GeomSequence tag="geom1"><lengthUnit>mm</lengthUnit></GeomSequence>
        <DatasetFeature tag="dset1" op="Solution"/>
        <TableFeature tag="tbl1"><realData>37.4</realData></TableFeature></Model>'''
        changed = xml.replace("1.43[T]", "1.40[T]")
        with tempfile.TemporaryDirectory() as directory:
            source_folder = Path(directory) / "source"; source_folder.mkdir()
            source = source_folder / "model.mph"
            nested = io.BytesIO()
            with zipfile.ZipFile(nested, "w") as archive:
                archive.writestr("dmodel.xml", changed)
            with zipfile.ZipFile(source, "w") as archive:
                archive.writestr("dmodel.xml", xml)
                archive.writestr("savepoint1/model.zip", nested.getvalue())
                archive.writestr("solutionstatic1.mphbin", b"test")
                archive.writestr("modelinfo.xml", '<modelInfo comsolVersion="6.2" nodeType="solved"/>')
            original = source.read_bytes()
            audit = inspect_mph(source)
            self.assertTrue(audit["source_unchanged"])
            self.assertIn("parameters", audit["current_vs_saved_snapshot_differences"]["savepoint1"])
            job = prepare_existing_solution_export(source, [[0, 0, 0], [5, 0, 0]],
                                                   Path(directory) / "job", Path(directory) / "comsol")
            java = Path(job["java_path"]).read_text(encoding="ascii")
            self.assertIn("ModelUtil.loadCopy", java)
            self.assertIn("-nosave", job["batch_command"])
            self.assertNotIn(".study(", java)
            self.assertNotIn(".mesh(", java)
            self.assertNotIn(".save(", java)
            self.assertEqual(source.read_bytes(), original)

    def test_comsol_java_output_in_separate_native_log_is_consumed(self):
        # COMSOL's progress stdout is not where Java System.out normally goes.
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory); source = folder / "original.mph"
            source.write_bytes(b"read-only source")
            executable = folder / "dummy.exe"; executable.write_bytes(b"not executed")
            logs = folder / "native_logs"; logs.mkdir()
            job = {"output_dir": str(folder), "source_mph": str(source),
                   "source_sha256": sha256_file(source), "mode": "existing_solution_interpolation_only",
                   "compile_command": [str(executable)], "batch_command": [str(executable)],
                   "point_count": 2, "points_mm": [[0, 0, 0], [10, 0, 0]],
                   "export_job_id": "test_unique_id", "comsol_log_dir": str(logs),
                   "output_csv": str(folder / "field.csv")}
            def successful_process(*args, **kwargs):
                (logs / "batch1.log").write_text(
                    "MRI_EXPORT_JOB_ID,test_unique_id\n"
                    "MRI_FIELD_ROW,0,0,0,0,0,0.04\n"
                    "MRI_FIELD_ROW,10,0,0,0,0,0.039\n", encoding="utf-8")
                return type("Result", (), {"returncode": 0})()
            with patch("mri_magnet.comsol_bridge.subprocess.run", side_effect=successful_process):
                result = run_existing_solution_export(job)
            self.assertTrue(result["source_unchanged"])
            self.assertEqual(read_field_csv(job["output_csv"])["fields_T"][1][2], .039)


if __name__ == "__main__":
    unittest.main()
