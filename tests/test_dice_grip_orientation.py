import contextlib
import io
import sys
import unittest
from unittest.mock import patch

import cv2
import numpy as np
import test_dice_area_pick as area
from test_dice_pick_place import pickup
with patch.object(sys, 'path', [str(area.detector.Path(__file__).resolve().parents[1] / 'experiments' / 'vision'), *sys.path]):
    import dice_grip_orientation as yaw


def box(angle):
    return cv2.boxPoints(((240,190),(24,24),float(angle)))


class OrientationTests(unittest.TestCase):
    def setUp(self):
        self.cal = pickup.PickCalibration(area.area_data())
        self.workspace = area.detector.detect_workspace(area.workspace_frame())
        self.cal.auto_yaw = True
        self.cal.yaw_reference = yaw.make_reference(self.cal,box(0),self.workspace,self.cal.pickup_reference)

    def test_world_rotation_preserves_xyz_and_tilt(self):
        pose = pickup.resolve_pick_target(self.cal,(240,190),(640,480),self.workspace,box=box(30))
        np.testing.assert_allclose(pose[:3],[.22,-.105,.2],atol=1e-6)
        np.testing.assert_allclose(pose[3:5],self.cal.pickup_reference[3:5])
        self.assertAlmostEqual(pose[5],np.deg2rad(30),places=5)

    def test_square_symmetry_avoids_ninety_degree_wrist_jumps(self):
        for angle in [0,90,180,270]:
            pose = pickup.resolve_pick_target(self.cal,(240,190),(640,480),self.workspace,box=box(angle))
            self.assertAlmostEqual(pose[5],0,places=5)
        pose = pickup.resolve_pick_target(self.cal,(240,190),(640,480),self.workspace,box=box(80))
        self.assertAlmostEqual(pose[5],np.deg2rad(-10),places=5)

    def test_camera_angles_are_mapped_into_robot_axes(self):
        self.cal.matrix[:2,:2] = [[0,.0005],[-.0005,0]]
        self.cal.yaw_reference = yaw.make_reference(self.cal,box(0),self.workspace,self.cal.pickup_reference)
        pose = pickup.resolve_pick_target(self.cal,(240,190),(640,480),self.workspace,box=box(20))
        self.assertAlmostEqual(pose[5],np.deg2rad(20),places=5)

    def test_reference_accounts_for_actual_reference_tool_rotation(self):
        actual = self.cal.pickup_reference.copy();actual[5]=.2
        self.cal.yaw_reference = yaw.make_reference(self.cal,box(0),self.workspace,actual)
        pose = pickup.resolve_pick_target(self.cal,(240,190),(640,480),self.workspace,box=box(10))
        self.assertAlmostEqual(pose[5],np.deg2rad(10)+.2,places=5)

    def test_missing_or_incompatible_reference_blocks_rotation(self):
        self.cal.yaw_reference = None
        with self.assertRaises(ValueError):
            pickup.resolve_pick_target(self.cal,(240,190),(640,480),self.workspace,box=box(0))
        self.cal.yaw_reference = yaw.make_reference(self.cal,box(0),self.workspace,self.cal.pickup_reference)
        self.cal.matrix[0,2]+=.01
        with self.assertRaises(ValueError):
            pickup.resolve_pick_target(self.cal,(240,190),(640,480),self.workspace,box=box(0))


class LowerPreviewTests(unittest.TestCase):
    def setUp(self):
        area.AreaMotionTests.setUp(self)

    def move(self,pose):
        self.pose=np.asarray(pose)

    def test_lowering_is_vertical_and_gripper_is_never_actuated(self):
        with (patch('builtins.input',side_effect=['PREVIEW','RETURN']),contextlib.redirect_stdout(io.StringIO())):
            self.assertTrue(pickup.preview_pick(self.robot,self.mapping,(242,190),(640,480),self.revalidate,
                                                workspace=self.workspace,clearance_mm=80))
        moves=[c.args[0]for c in self.robot.move_linear_pose.call_args_list]
        self.assertEqual(len(moves),4)
        np.testing.assert_allclose(moves[0][:2],moves[1][:2])
        self.assertAlmostEqual(moves[1][2],.28)
        self.assertAlmostEqual(moves[2][2],.4)
        np.testing.assert_allclose(moves[-1],self.mapping.overview)
        self.robot.open_gripper.assert_not_called()
        self.robot.close_gripper.assert_not_called()

    def test_thirty_mm_preview_clearance_is_above_grip_pose_without_gripper_actuation(self):
        with (patch('builtins.input',side_effect=['PREVIEW','RETURN']),contextlib.redirect_stdout(io.StringIO())):
            self.assertTrue(pickup.preview_pick(self.robot,self.mapping,(242,190),(640,480),self.revalidate,
                                                workspace=self.workspace,clearance_mm=30))
        moves=[c.args[0] for c in self.robot.move_linear_pose.call_args_list]
        self.assertAlmostEqual(moves[1][2],.23)
        self.robot.open_gripper.assert_not_called()
        self.robot.close_gripper.assert_not_called()

    def test_fifty_percent_upper_approach_uses_twenty_for_lower_rise_return(self):
        speed = [None]
        movement_speeds = []
        self.robot.set_arm_max_velocity.side_effect = lambda value: speed.__setitem__(0,value)
        def move(pose):
            movement_speeds.append(speed[0])
            self.move(pose)
        self.robot.move_linear_pose.side_effect = move
        with (patch('builtins.input',side_effect=['PREVIEW','RETURN']),contextlib.redirect_stdout(io.StringIO())):
            self.assertTrue(pickup.preview_pick(self.robot,self.mapping,(242,190),(640,480),self.revalidate,
                                                workspace=self.workspace,clearance_mm=30,speed_percent=50))
        self.assertEqual(movement_speeds,[50,20,20,20])
        self.assertEqual(speed[0],10)
        self.robot.close_gripper.assert_not_called()

    def test_invalid_preview_speed_never_actuates(self):
        for speed in (0,51,100,float('nan'),True):
            with self.assertRaises(ValueError):
                pickup.preview_pick(self.robot,self.mapping,(242,190),(640,480),self.revalidate,
                                    workspace=self.workspace,speed_percent=speed)
        self.robot.set_arm_max_velocity.assert_not_called()
        self.robot.move_linear_pose.assert_not_called()

    def test_cancel_at_low_pose_does_not_make_recovery_move(self):
        with (patch('builtins.input',side_effect=['PREVIEW','STOP']),contextlib.redirect_stdout(io.StringIO())):
            with self.assertRaisesRegex(ValueError,'inspection height'):
                pickup.preview_pick(self.robot,self.mapping,(242,190),(640,480),self.revalidate,
                                     workspace=self.workspace,clearance_mm=80)
        self.assertEqual(self.robot.move_linear_pose.call_count,2)
        self.robot.close_gripper.assert_not_called()

    def test_too_low_or_above_overview_clearance_is_rejected_before_actuation(self):
        for clearance in [0,20,float('nan'),300]:
            with self.assertRaises(ValueError):
                pickup.preview_pick(self.robot,self.mapping,(242,190),(640,480),self.revalidate,
                                     workspace=self.workspace,clearance_mm=clearance)
        self.robot.move_linear_pose.assert_not_called()


class LiveAlignmentTests(unittest.TestCase):
    setUp = LowerPreviewTests.setUp
    move = LowerPreviewTests.move

    def test_selectable_steps_preserve_height_without_total_ten_mm_limit(self):
        origin = np.array([.2,.0,.16,-.15,1.5,.3])
        for step in (1,2,5):
            target = pickup.alignment_step(origin,origin,0,1,step_mm=step)
            self.assertAlmostEqual(target[0],.2+step/1000)
            np.testing.assert_array_equal(target[1:],origin[1:])
        target=origin.copy()
        for _ in range(4):
            target=pickup.alignment_step(origin,target,1,1)
        self.assertAlmostEqual(target[1],.020)
        for axis,step in ((2,5),(0,0),(0,10),(0,float('nan')),(0,True)):
            with self.assertRaises(ValueError):
                pickup.alignment_step(origin,origin,axis,1,step_mm=step)

    def test_corrected_inspection_rises_vertically_without_gripping(self):
        def inspect(pose):
            corrected=pose.copy();corrected[0]+=.015;corrected[1]-=.001
            self.move(corrected)
            self.robot.set_arm_max_velocity(10)
            return corrected
        with patch('builtins.input',return_value='PREVIEW'),contextlib.redirect_stdout(io.StringIO()):
            pickup.preview_pick(self.robot,self.mapping,(242,190),(640,480),self.revalidate,
                                workspace=self.workspace,clearance_mm=30,inspection_callback=inspect)
        moves=[c.args[0] for c in self.robot.move_linear_pose.call_args_list]
        np.testing.assert_allclose(moves[2][:2],np.asarray(moves[1][:2])+[.015,-.001])
        self.assertAlmostEqual(moves[2][2],.4)
        self.robot.open_gripper.assert_not_called()
        self.robot.close_gripper.assert_not_called()

    def test_inspection_stop_does_not_rise_or_return(self):
        def stop(pose):
            raise ValueError('operator stop')
        with patch('builtins.input',return_value='PREVIEW'),contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(ValueError,'operator stop'):
                pickup.preview_pick(self.robot,self.mapping,(242,190),(640,480),self.revalidate,
                                    workspace=self.workspace,clearance_mm=30,inspection_callback=stop)
        self.assertEqual(self.robot.move_linear_pose.call_count,2)

    def test_camera_failure_blocks_live_nudge(self):
        with patch.object(pickup,'require_live_camera',side_effect=ValueError('camera frozen')):
            with self.assertRaisesRegex(ValueError,'camera frozen'):
                pickup.live_inspection(self.robot,np.array([.2,0,.23,0,1.5,0]),None,'test')
        self.robot.move_linear_pose.assert_not_called()
        self.robot.set_arm_max_velocity.assert_not_called()


    def test_alignment_wait_retry_never_resends_movement(self):
        with (patch.object(pickup,'wait_for_pose',side_effect=[ValueError('orientation not settled'),None]) as wait,
              patch('builtins.input',return_value='WAIT'),contextlib.redirect_stdout(io.StringIO())):
            pickup.wait_for_preview_pose(self.robot,self.mapping.overview)
        self.assertEqual(wait.call_count,2)
        self.robot.move_linear_pose.assert_not_called()
