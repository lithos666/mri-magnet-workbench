"""Source-slot identity and physical displacements in reference layout drawings."""
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

from PIL import Image
from mri_magnet.geometry import Design, RingSpec
from mri_magnet.layout import layout_manifest, layout_comparison_manifest, render_rz_layout


class LayoutComparisonTests(unittest.TestCase):
    def fixture(self):
        return Design((RingSpec('A0',0,0,128.,40.,28,'out',0.),
                       RingSpec('A2',0,2,156.,40.,28,'down',0.),
                       RingSpec('A4',0,4,184.,40.,28,'in',0.),
                       RingSpec('N0',13,0,128.,-40.,28,'in',0.)),name='reference')

    def test_added_removed_source_slots_are_not_matched_by_radial_rank(self):
        seed=self.fixture()
        candidate=replace(seed,rings=(replace(seed.rings[0],radius_mm=129.),
            RingSpec('restored3',0,3,170.,40.,32,'out',0.),
            replace(seed.rings[2],radius_mm=185.,id='renamed4'),seed.rings[3]))
        comparison=layout_comparison_manifest(candidate,seed)
        rows={(r['layer_index'],r['slot_index']):r for r in comparison['rings']}
        self.assertEqual(rows[(0,2)]['status'],'removed')
        self.assertEqual(rows[(0,3)]['status'],'added')
        self.assertIsNone(rows[(0,3)]['delta_r_mm'])
        self.assertEqual(rows[(0,4)]['status'],'matched')
        self.assertEqual(rows[(0,4)]['delta_r_mm'],1.)
        self.assertEqual(comparison['added_count'],1)
        self.assertEqual(comparison['removed_count'],1)
        self.assertEqual(comparison['layers'][0]['current_magnet_count'],88)

    def test_radial_and_axial_signed_changes_remain_in_mm(self):
        seed=self.fixture()
        candidate=replace(seed,rings=tuple(replace(r,radius_mm=r.radius_mm-1.8,
            z_mm=r.z_mm-2.1 if r.z_mm>0 else r.z_mm+2.1) for r in seed.rings))
        comparison=layout_comparison_manifest(candidate,seed)
        positive=comparison['matched'][0];negative=comparison['matched'][-1]
        self.assertAlmostEqual(positive['delta_r_mm'],-1.8)
        self.assertAlmostEqual(positive['delta_z_mm'],-2.1)
        self.assertAlmostEqual(negative['delta_z_mm'],2.1)
        self.assertAlmostEqual(positive['displacement_mm'],(1.8**2+2.1**2)**.5)

    def test_identical_layout_has_no_false_moves_or_schema_changes(self):
        seed=self.fixture();before=layout_manifest(seed)
        comparison=layout_comparison_manifest(seed,seed)
        self.assertEqual(comparison['moved_count'],0)
        self.assertTrue(all(r['delta_r_mm']==0 and r['delta_z_mm']==0 for r in comparison['matched']))
        self.assertEqual(before,layout_manifest(seed))

    def test_duplicate_source_slot_is_rejected_even_with_different_display_ids(self):
        seed=self.fixture()
        candidate=replace(seed,rings=(seed.rings[0],replace(seed.rings[0],id='different_id')))
        with self.assertRaisesRegex(ValueError,'Duplicate source layer/slot'):
            layout_comparison_manifest(candidate,seed)

    def test_reference_overlay_renders_png_and_pdf_with_deleted_and_restored_slots(self):
        seed=self.fixture()
        candidate=replace(seed,rings=(replace(seed.rings[0],radius_mm=129.8,z_mm=35.),
            RingSpec('new3',0,3,171.,35.,28,'up',0.),replace(seed.rings[-1],z_mm=-35.)))
        with tempfile.TemporaryDirectory() as directory:
            png=render_rz_layout(candidate,Path(directory)/'compare.png',reference_design=seed)
            with Image.open(png) as image:
                self.assertEqual(image.size,(2560,1000))
                self.assertGreater(len(image.getcolors(2560*1000) or []),30)
            pdf=render_rz_layout(candidate,Path(directory)/'compare.pdf',reference_design=seed)
            self.assertTrue(pdf.read_bytes().startswith(b'%PDF-'))
        with self.assertRaises(ValueError):
            render_rz_layout(seed,'invalid.png',reference_design=seed,displacement_threshold_mm=-1.)


if __name__=='__main__':unittest.main()
