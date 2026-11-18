import copy
import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from test_dice_pick_place import pickup
from test_dice_area_pick import area_data
with patch.object(sys,'path',[str(Path(__file__).resolve().parents[1]/'experiments'/'vision'),*sys.path]):
    import dice_alignment_calibration as learning
    import dice_grip_orientation as yaw


class AlignmentLearningTests(unittest.TestCase):
    def setUp(self):
        self.cal=pickup.PickCalibration(area_data())
        self.cal.yaw_reference=dict(version=1,edge_angle=0.,actual_pose=self.cal.pickup_reference.tolist(),
                                   matrix=self.cal.matrix.tolist(),pickup_reference=self.cal.pickup_reference.tolist(),tool=self.cal.tool)
        self.offset=np.array([.002,-.001,.003,-.002])
        coords=[(.17,-.13),(.29,-.13),(.29,-.02),(.17,-.02),(.20,-.10),(.25,-.07),(.21,-.04),(.26,-.11)]
        angles=[-.6,.6,-.5,.5,-.3,.3,-.2,.2]
        self.data=dict(version=1,binding=learning.binding(self.cal),samples=[])
        for (x,y),angle in zip(coords,angles):
            a=np.array([x,y,.23,0.,1.57,angle])
            b=a.copy();b[:2]+=learning.design(angle)@self.offset;b[5]+=.04
            self.data['samples'].append(dict(planned=a.tolist(),corrected=b.tolist(),actual=b.tolist()))

    def test_recovers_offsets_and_predicts_new_position_and_rotation(self):
        model=learning.fit_samples(self.data)
        np.testing.assert_allclose(model['offset'],self.offset,atol=1e-10)
        self.assertAlmostEqual(model['turn'],.04)
        self.cal.alignment_model=model
        target=np.array([.23,-.08,.2,0.,1.57,.1])
        corrected=learning.apply_alignment(self.cal,target)
        np.testing.assert_allclose(corrected[:2],target[:2]+learning.design(.1)@self.offset)
        self.assertAlmostEqual(corrected[5],.14)
        np.testing.assert_array_equal(corrected[2:5],target[2:5])
        self.assertLess(model['max_error_mm'],1e-6)

    def test_insufficient_samples_geometry_and_angles_rejected(self):
        bad=copy.deepcopy(self.data);bad['samples']=bad['samples'][:7]
        with self.assertRaisesRegex(ValueError,'at least 8'):learning.fit_samples(bad)
        for key in ('position','angle'):
            bad=copy.deepcopy(self.data)
            for s in bad['samples']:
                if key=='position':
                    delta=np.array(s['corrected'])[:2]-np.array(s['planned'])[:2]
                    s['planned'][:2]=[.2,-.1];s['corrected'][:2]=(np.array([.2,-.1])+delta).tolist()
                else:s['planned'][5]=0.;s['corrected'][5]=.04
            with self.assertRaises(ValueError):learning.fit_samples(bad)

    def test_inconsistent_sample_rejected_by_held_out_prediction(self):
        for kind in ('xy','yaw'):
            bad=copy.deepcopy(self.data)
            if kind=='xy':bad['samples'][3]['corrected'][0]+=.009
            else:bad['samples'][3]['corrected'][5]+=.1
            with self.assertRaisesRegex(ValueError,'leave-one-out'):learning.fit_samples(bad)

    def test_extrapolation_and_changed_binding_blocked(self):
        self.cal.alignment_model=learning.fit_samples(self.data)
        for target in ([.32,-.08,.2,0,1.57,0],[.23,-.08,.2,0,1.57,.9]):
            with self.assertRaisesRegex(ValueError,'Outside learned'):learning.apply_alignment(self.cal,target)
        self.cal.matrix[0,2]+=.001
        with self.assertRaisesRegex(ValueError,'binding'):learning.apply_alignment(self.cal,[.23,-.08,.2,0,1.57,0])

    def test_samples_accumulate_atomically_and_different_binding_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp,contextlib.redirect_stdout(io.StringIO()):
            path=Path(temp)/'examples.json'
            for s in self.data['samples'][:2]:
                report=dict(planned=s['planned'],corrected_command=s['corrected'],corrected_actual=s['actual'])
                learning.append_sample(self.cal,report,path)
            self.assertEqual(len(learning.load_samples(self.cal,path)['samples']),2)
            with self.assertRaisesRegex(ValueError,'too similar'):learning.append_sample(self.cal,report,path)
            self.cal.pickup_reference[0]+=.001
            with self.assertRaisesRegex(ValueError,'different'):learning.load_samples(self.cal,path)
            self.assertEqual(len(json.loads(path.read_text())['samples']),2)

    def test_manual_yaw_step_preserves_position_and_respects_limit(self):
        a=np.array([.2,0,.16,-.15,1.5,.3])
        b=pickup.alignment_turn(a,a,1)
        np.testing.assert_array_equal(a[:5],b[:5])
        self.assertAlmostEqual(b[5]-a[5],np.deg2rad(2))
        for _ in range(6):b=pickup.alignment_turn(a,b,1)
        with self.assertRaises(ValueError):pickup.alignment_turn(a,b,1)

    def test_height_or_large_correction_not_learned(self):
        a=np.array([.2,0,.16,0,1.57,0])
        for index,delta in ((2,.001),(0,.03),(5,.4)):
            b=a.copy();b[index]+=delta
            with self.assertRaises(ValueError):learning.check_pair(a,b)


    def test_automatic_correction_is_applied_once_after_mapping(self):
        from test_dice_area_pick import detector,workspace_frame
        workspace=detector.detect_workspace(workspace_frame())
        target=pickup.resolve_pick_target(self.cal,(240,190),(640,480),workspace)
        self.cal.alignment_model=learning.fit_samples(self.data)
        actual=pickup.resolve_pick_target(self.cal,(240,190),(640,480),workspace)
        expected=learning.apply_alignment(self.cal,target)
        np.testing.assert_allclose(actual,expected)
        again=pickup.resolve_pick_target(self.cal,(240,190),(640,480),workspace)
        np.testing.assert_allclose(again,expected)

    def test_changed_inspection_height_does_not_pollute_saved_examples(self):
        with tempfile.TemporaryDirectory() as temp,contextlib.redirect_stdout(io.StringIO()):
            path=Path(temp)/'examples.json'
            first=self.data['samples'][0]
            report=dict(planned=first['planned'],corrected_command=first['corrected'],corrected_actual=first['actual'])
            learning.append_sample(self.cal,report,path)
            bad=copy.deepcopy(report);bad['planned'][0]+=.02;bad['corrected_command'][0]+=.02
            bad['planned'][2]+=.001;bad['corrected_command'][2]+=.001
            with self.assertRaisesRegex(ValueError,'height differs'):learning.append_sample(self.cal,bad,path)
            self.assertEqual(len(learning.load_samples(self.cal,path)['samples']),1)
