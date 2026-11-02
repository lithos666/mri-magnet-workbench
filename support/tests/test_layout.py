"""Placement drawings retain exact candidate coordinates and arrow semantics."""
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from matplotlib.figure import Figure

from PIL import Image
from mri_magnet.geometry import Design, RingSpec
from mri_magnet.layout import layout_manifest, render_rz_layout


class LayoutTests(unittest.TestCase):
    def fixture(self):
        return Design((
            RingSpec("R01-L1", 0, 0, 128.375, 247.0, 28, "out", 276.428571),
            RingSpec("R01-L3", 0, 2, 154.4, 247.0, 28, "down", 276.428571),
            RingSpec("R02-L1", 1, 0, 130.0, 208.25, 32, "in", 275.625),
            RingSpec("R14-L4", 13, 3, 180.625, -246.8, 24, "up", 277.5),
        ), name="candidate_with_changed_counts")

    def test_manifest_preserves_candidate_coordinates_sparse_ids_and_counts(self):
        design = self.fixture();manifest = layout_manifest(design)
        by_id = {row["id"]: row for row in manifest["rings"]}
        self.assertEqual(by_id["R01-L1"]["center_r_mm"], 128.375)
        self.assertEqual(by_id["R02-L1"]["center_z_mm"], 208.25)
        self.assertEqual(by_id["R14-L4"]["slot_index"], 3)
        self.assertEqual(by_id["R14-L4"]["azimuth_count"], 24)
        self.assertEqual(manifest["magnet_count"], 112)
        self.assertEqual([x["magnet_count"] for x in manifest["layers"]], [56, 32, 24])
        self.assertFalse(manifest["common_column_phase"])
        self.assertEqual(manifest["positive_z_display_side"], "left")

    def test_display_radius_is_inner_face_and_keeps_model_center(self):
        design = self.fixture()
        before = design.to_dict()
        manifest = layout_manifest(design)
        row = manifest["rings"][0]
        self.assertEqual(row["center_r_mm"], 128.375)
        self.assertEqual(row["display_radius_mm"], 122.375)
        self.assertEqual(row["display_radius_mm"], row["r_min_mm"])
        self.assertEqual(manifest["radial_label_reference"], "inner_radial_face")
        self.assertEqual(manifest["radial_label_formula"], "center_r_mm - cube_mm / 2")
        self.assertEqual(design.to_dict(), before)
        smaller = layout_manifest(replace(design, cube_mm=10.0))["rings"][0]
        self.assertEqual(smaller["center_r_mm"], 128.375)
        self.assertEqual(smaller["display_radius_mm"], 123.375)

    def test_render_labels_use_bottom_edge_and_explain_coordinates(self):
        with tempfile.TemporaryDirectory() as folder:
            with patch.object(Figure, "savefig", autospec=True) as saved:
                render_rz_layout(self.fixture(), Path(folder)/"layout.png")
            figure = saved.call_args.args[0]
            block_labels = [text.get_text() for text in figure.axes[0].texts]
            notes = [text.get_text() for text in figure.texts]
        self.assertIn("122.4", block_labels)
        self.assertNotIn("128.4", block_labels)
        self.assertTrue(any("底边半径 R底=R中心−边长/2" in text for text in notes))
        self.assertTrue(any("Z为中心坐标" in text for text in notes))

    def test_arrow_semantics_account_for_positive_z_on_left(self):
        rows = {x["polarity"]: x for x in layout_manifest(self.fixture())["rings"]}
        self.assertEqual(rows["out"]["display_arrow"], "↑")
        self.assertEqual(rows["in"]["display_arrow"], "↓")
        self.assertEqual(rows["down"]["display_arrow"], "→")
        self.assertEqual(rows["up"]["display_arrow"], "←")
        self.assertEqual(rows["down"]["north_rz"], [0.0, -1.0])
        self.assertEqual(rows["up"]["north_rz"], [0.0, 1.0])

    def test_nonuniform_spacing_reported_as_actual_intervals(self):
        manifest = layout_manifest(self.fixture())
        self.assertEqual(manifest["axial_center_pitches_mm"], [455.05, 38.75])
        self.assertAlmostEqual(manifest["axial_span_mm"], 505.8)
        self.assertAlmostEqual(manifest["minimum_same_column_radial_face_gap_mm"], 14.025)

    def test_png_and_vector_pdf_are_created_from_the_actual_design(self):
        with tempfile.TemporaryDirectory() as folder:
            target = render_rz_layout(self.fixture(), Path(folder)/"layout.png", manufacturing_audit={"valid": True})
            with Image.open(target) as image:
                self.assertEqual(image.size, (2560, 1000))
                self.assertGreater(len(image.getcolors(2560*1000) or []), 20)
            pdf = render_rz_layout(self.fixture(), Path(folder)/"layout.pdf")
            self.assertTrue(pdf.read_bytes().startswith(b"%PDF-"))

    def test_inconsistent_layer_z_cannot_be_silently_drawn_as_one_column(self):
        d = self.fixture()
        wrong = replace(d, rings=(d.rings[0], replace(d.rings[1], z_mm=246.0)))
        with self.assertRaises(ValueError):layout_manifest(wrong)


if __name__ == "__main__":
    unittest.main()
