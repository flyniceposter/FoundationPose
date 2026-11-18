"""CPU tests for perturbation geometry, bootstrap pairing and selection isolation."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import mask_study as m


class MaskStudyTests(unittest.TestCase):
    def test_geometry_and_clipping(self):
        mask = np.zeros((30, 30), bool); mask[8:22, 8:22] = True
        self.assertTrue(np.array_equal(mask, m.perturb(mask, 'control')))
        self.assertTrue(np.all(m.perturb(mask, 'erode2') <= mask))
        self.assertTrue(np.all(mask <= m.perturb(mask, 'dilate5')))
        point = np.zeros((10, 10), bool); point[4, 3] = True
        expected = np.zeros_like(point); expected[4, 5] = True
        self.assertTrue(np.array_equal(m.perturb(point, 'shiftx+2'), expected))
        self.assertFalse(m.perturb(point, 'shiftx-5').any())
        expected[:] = False; expected[2, 3] = True
        self.assertTrue(np.array_equal(m.perturb(point, 'shifty-2'), expected))
        self.assertFalse(m.perturb(point, 'erode2').any())

    def test_repair_no_input_mutation(self):
        mask = np.zeros((30, 30), bool); mask[5:25, 5:25] = True; mask[15, 15] = False
        prior = mask.copy()
        self.assertTrue(m.repair(mask, 'close2')[15, 15])
        for method in m.METHODS:
            result = m.repair(mask, method)
            self.assertEqual(result.dtype, bool)
            self.assertTrue(np.array_equal(mask, prior))

    def test_paired_bootstrap_preserves_replicates(self):
        raw = [dict(obj_id=o, im_id=i, condition=c, seed=s, adds_pass=0) for o in [1, 2] for i in [3, 4] for c in ['control', 'erode2'] for s in m.SEEDS]
        treated = [dict(r, adds_pass=1) for r in raw]
        delta, bounds = m.paired_interval(raw, treated, 'adds_pass')
        self.assertEqual(delta, 1)
        self.assertEqual(bounds, [1, 1])

    def test_selection_never_reads_heldout_and_freezes(self):
        def development_only(out, records, split, method):
            self.assertEqual(split, 'development')
            return [dict(obj_id=1, adds_pass=int(method=='close2'))]
        with tempfile.TemporaryDirectory() as d, patch.object(m, 'load_group', development_only):
            self.assertEqual(m.select(Path(d), []), 'close2')
            self.assertEqual(json.loads((Path(d)/'frozen_selection.json').read_text())['method'], 'close2')

    def test_complete_report_and_missing_checkpoint_rejected(self):
        records = [dict(obj_id=str(o), im_id=str(i), split=split) for o in [1, 2] for split, i in [('development', 3), ('held_out', 4)]]
        with tempfile.TemporaryDirectory() as d, patch.object(m, 'CONDITIONS', ['control', 'erode2']), patch.object(m, 'SEEDS', [0]):
            out = Path(d); (out/'checkpoints').mkdir()
            for row in records:
                for method in m.METHODS:
                    if row['split']=='held_out' and method!='raw':
                        continue
                    for condition in m.CONDITIONS:
                        value = dict(obj_id=int(row['obj_id']), im_id=int(row['im_id']), split=row['split'],
                            condition=condition, method=method, seed=0, status='valid', adds_pass=1,
                            mssd_pass=1, mspd_pass=1, warmed=True, latency_s=.5, mask_iou=1.,
                            translation_mm=1., rotation_deg=1., add_mm=1., adi_mm=1.)
                        m.atomic_json(m.checkpoint(out, row, condition, 0, method), value)
            m.report(out, records)
            self.assertEqual(json.loads((out/'status.json').read_text())['state'], 'complete')
            self.assertTrue((out/'heldout_recall.png').exists())
            self.assertTrue((out/'per_object_summary.csv').exists())
            m.checkpoint(out, records[-1], 'control', 0, 'raw').unlink()
            with self.assertRaisesRegex(RuntimeError, 'Incomplete stage'):
                m.report(out, records)


if __name__ == '__main__':
    unittest.main()
