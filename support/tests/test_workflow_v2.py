"""CLI configuration reaches the same geometry-only workflow in both modes."""
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from mri_magnet import cli, workflow
from mri_magnet.geometry import Design


class UnifiedCheckWorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root=Path(__file__).resolve().parents[2]
        cls.source=Path(os.environ.get("MRI_SOURCE",str(root/"examples"/"seeded"/"placement.json")))
        if not cls.source.is_file():raise unittest.SkipTest("Set MRI_SOURCE to the actual source geometry")

    def test_check_honours_all_configuration_without_field_calls_in_both_modes(self):
        for enabled in (True,False):
            with self.subTest(manufacturing_enabled=enabled),tempfile.TemporaryDirectory() as folder:
                directory=Path(folder);out=directory/"check_run"
                overrides={"manufacturing":{"enabled":enabled},
                    "source_overrides":{"enabled":True,"cube_mm":11.0,"remanence_T":1.39,"reference_mur":1.07},
                    "envelope":{"protected_bore_diameter_mm":250.0,"max_outer_diameter_mm":400.0,"max_length_mm":520.0},
                    "sampling":{"quick_training_area":16,"quick_training_boundary":20,
                                "quick_validation_area":256,"quick_validation_boundary":192}}
                config=directory/"overrides.json"
                config.write_text(json.dumps(overrides),encoding="utf-8")
                original_workflow=workflow.run_manufacturing_workflow
                with patch.object(workflow,"run_manufacturing_workflow",wraps=original_workflow) as unified, \
                     patch.object(workflow.FieldEngine,"field",side_effect=AssertionError("check must never calculate magnetic fields")) as field, \
                     patch.object(cli,"log"):
                    code=cli.main(["check","--quick","--config",str(config),"--source",str(self.source),
                                   "--mph",str(directory/"absent.mph"),"--out",str(out)])
                    self.assertEqual(code,0)
                    unified.assert_called_once()
                    field.assert_not_called()
                details=out/"详细数据"
                snapshot=json.loads((details/"run_config.json").read_text(encoding="utf-8"))
                configuration=snapshot["configuration"]
                self.assertEqual(configuration["manufacturing"]["enabled"],enabled)
                self.assertEqual(configuration["sampling"]["quick_training_area"],16)
                self.assertEqual(configuration["sampling"]["quick_training_boundary"],20)
                self.assertEqual(configuration["sampling"]["quick_validation_area"],256)
                self.assertEqual(configuration["sampling"]["quick_validation_boundary"],192)
                self.assertEqual(snapshot["geometry_limits"]["bore_radius_mm"],125.0)
                self.assertEqual(snapshot["geometry_limits"]["outer_radius_mm"],200.0)
                self.assertEqual(snapshot["geometry_limits"]["half_length_mm"],260.0)
                seed=Design.from_dict(json.loads((details/"manufacturing_seed_design.json").read_text(encoding="utf-8")))
                self.assertEqual(seed.cube_mm,11.0)
                self.assertEqual(seed.remanence_T,1.39)
                self.assertEqual(seed.reference_mur,1.07)
                self.assertGreaterEqual(2*min(r.radius_mm-seed.cube_mm/2 for r in seed.rings),250.0-1e-6)
                summary=json.loads((details/"constraint_summary.json").read_text(encoding="utf-8"))
                self.assertTrue(summary["seed_valid"])
                space=json.loads((details/"search_space.json").read_text(encoding="utf-8"))
                self.assertTrue(space["construction_metadata"]["fixed_topology"])
                self.assertTrue((out/"约束检查.txt").is_file())


if __name__=="__main__":unittest.main()
