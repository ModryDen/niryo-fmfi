import contextlib
import io
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest
from unittest.mock import Mock, patch
import numpy as np
from pyniryo import ToolID
from test_dice_camera_calibration import detector, sample_data, workspace_frame, MODULE_PATH
with patch.object(sys, "path", [str(MODULE_PATH.parent), *sys.path]):
    from dice_pick_place import PickCalibration, pick_and_place, save_pick_draft
    import dice_pick_place as pickup
    from dice_camera_calibration import same_orientation


class PickPlaceTests(unittest.TestCase):
    def setUp(self):
        data = sample_data()
        data.update(purpose='dice_pick_place', drop=[.3, .1, .22, 0, 1.57, 0],
                    board_corners=[[100, 80], [540, 80], [540, 420], [100, 420]], tool='GRIPPER_1')
        self.mapping = PickCalibration(data)
        self.robot = Mock()
        self.pose = self.mapping.overview.copy()
        self.robot.get_pose.return_value.to_list.side_effect = lambda: self.pose.tolist()
        self.robot.get_current_tool_id.return_value = ToolID.GRIPPER_1
        self.robot.get_learning_mode.return_value = False
        self.robot.move_linear_pose.side_effect = self.move
        self.robot.get_hardware_status.return_value = SimpleNamespace(
            connection_up=True, calibration_needed=False, calibration_in_progress=False,
            motor_names=['joint_1', 'Tool'], hardware_errors=[0, 0], error_message='')
        self.check = Mock()

    def move(self, pose):
        self.pose = np.asarray(pose)

    def run_pick(self, answer='PICK'):
        with patch('builtins.input', return_value=answer), contextlib.redirect_stdout(io.StringIO()):
            return pick_and_place(self.robot, self.mapping, (300, 250), (640, 480), self.check)

    def assert_no_actuation(self):
        self.robot.move_linear_pose.assert_not_called()
        self.robot.open_gripper.assert_not_called()
        self.robot.close_gripper.assert_not_called()
        self.robot.set_arm_max_velocity.assert_not_called()

    def test_equivalent_euler_angles_near_vertical(self):
        first = [0, 0, 0, .1, np.pi / 2, .2]
        equivalent = [0, 0, 0, 1.1, np.pi / 2, 1.2]
        self.assertTrue(same_orientation(first, equivalent))
        # Crossing the Euler singularity can change roll/yaw by pi.
        first = [0, 0, 0, .1, np.pi / 2 - .001, .2]
        equivalent = [0, 0, 0, .1 + np.pi, np.pi / 2 + .001, .2 + np.pi]
        self.assertTrue(same_orientation(first, equivalent))
        different = [0, 0, 0, .1, np.pi / 2 - .001, .3]
        self.assertFalse(same_orientation(first, different))

    def test_draft_preserves_rejected_calibration(self):
        data = sample_data()
        data.update(purpose='dice_pick_place', poses=[list(p) for p in data['poses']])
        data['poses'][2][4] += .1
        with tempfile.TemporaryDirectory() as directory:
            draft = Path(directory) / 'draft.json'
            with patch.object(pickup, 'PICK_DRAFT_FILE', draft):
                save_pick_draft(data)
                with self.assertRaisesRegex(ValueError, 'Target 3'):
                    PickCalibration(data)
                self.assertEqual(json.loads(draft.read_text()), data)
                data['poses'].pop()
                save_pick_draft(data)
                self.assertEqual(len(json.loads(draft.read_text())['poses']), 4)

    def test_replace_only_one_saved_target(self):
        data = sample_data()
        corrected = list(data['poses'][2])
        corrected[0] += .001
        result = pickup.replace_pick_record(data, '3', corrected)
        self.assertEqual(result['poses'][2], corrected)
        self.assertNotEqual(result['poses'][2], data['poses'][2])
        self.assertEqual(result['pixels'], data['pixels'])
        self.assertEqual(result['poses'][:2], data['poses'][:2])
        self.assertEqual(result['poses'][3:], data['poses'][3:])

    def test_repair_rejects_bad_angle_and_height_without_mutation(self):
        data = sample_data()
        original = json.dumps(data)
        for component, difference in [(3, .2), (2, .01)]:
            pose = list(data['poses'][0])
            pose[component] += difference
            with self.assertRaises(ValueError):
                pickup.replace_pick_record(data, '1', pose)
            self.assertEqual(json.dumps(data), original)

    def test_image_repair_preserves_all_robot_records(self):
        data = sample_data()
        data['drop'] = [.3, .1, .2, 0, 1.57, 0]
        original = json.dumps(data)
        pixels = np.asarray(data['pixels']) + [2, 3]
        result = pickup.replace_pick_pixels(data, pixels)
        self.assertEqual(result['poses'], data['poses'])
        self.assertEqual(result['drop'], data['drop'])
        self.assertEqual(result['overview'], data['overview'])
        self.assertEqual(result['pixels'], pixels.tolist())
        self.assertEqual(json.dumps(data), original)

    def test_image_repair_rejects_invalid_geometry(self):
        data = sample_data()
        for points in (data['pixels'][:4], [[0, 0]] * 5,
                       [[100,100],[500,100],[500,400],[100,400],[600,450]],
                       [[100,100],[500,100],[500,400],[100,400],[float('nan'),200]]):
            with self.assertRaises(ValueError):
                pickup.replace_pick_pixels(data, points)

    def test_image_repair_at_wrong_pose_never_opens_camera(self):
        data = sample_data()
        with patch.object(pickup, 'read_frame') as read:
            self.pose[0] += .02
            with self.assertRaises(ValueError):
                pickup.reselect_pick_pixels(self.robot, None, data)
            read.assert_not_called()
        self.assert_no_actuation()

    def test_view_at_wrong_pose_never_saves_or_actuates(self):
        data = sample_data()
        original = json.dumps(data)
        self.pose[0] += .02
        with (patch.object(pickup, 'read_frame', return_value=workspace_frame()),
              patch.object(pickup.cv2, 'namedWindow'),
              patch.object(pickup.cv2, 'imshow'),
              patch.object(pickup.cv2, 'waitKey', return_value=ord('q')),
              patch.object(pickup.cv2, 'destroyWindow'),
              patch.object(pickup, 'save_pick_draft') as save,
              contextlib.redirect_stdout(io.StringIO()) as output):
            pickup.view_pick_draft(self.robot, None, data)
        self.assertIn('20.00 mm', output.getvalue())
        self.assertIn('orientation difference', output.getvalue())
        save.assert_not_called()
        self.assertEqual(json.dumps(data), original)
        self.assert_no_actuation()

    def test_board_diagnostics_distinguish_detection_failure_and_drift(self):
        workspace = detector.detect_workspace(workspace_frame())
        reference = workspace.corners.copy()
        pickup.require_board_reference(workspace, reference)
        with self.assertRaisesRegex(ValueError, 'detection failed'):
            pickup.require_board_reference(None, reference)
        workspace.corners += [4, 0]
        with self.assertRaisesRegex(ValueError, 'all 4 markers') as error:
            pickup.require_board_reference(workspace, reference)
        self.assertIn('[4.0, 4.0, 4.0, 4.0]', str(error.exception))
        np.testing.assert_array_equal(reference, [[100,80],[540,80],[540,420],[100,420]])

    def test_new_image_points_save_board_reference_from_same_frame(self):
        data = sample_data()
        workspace = detector.detect_workspace(workspace_frame())
        data['board_corners'] = (workspace.corners + [2, 0]).tolist()
        old = json.dumps(data)
        points = [[150,120],[490,120],[490,380],[150,380],[320,250]]
        result = pickup.replace_pick_pixels(data, points, workspace)
        self.assertEqual(result['board_corners'], workspace.corners.tolist())
        self.assertNotEqual(result['board_corners'], data['board_corners'])
        self.assertEqual(result['poses'], data['poses'])
        self.assertEqual(result['overview'], data['overview'])
        self.assertEqual(json.dumps(data), old)
        # A genuinely shifted board is still rejected; the tolerance is not relaxed.
        workspace.corners += [10, 0]
        with self.assertRaises(ValueError):
            pickup.replace_pick_pixels(data, points, workspace)

    def test_preflight_rejection_is_distinct_from_motion_failure(self):
        self.check.side_effect = ValueError('Board drift')
        with self.assertRaises(pickup.PickupPreflightError):
            self.run_pick()
        self.assert_no_actuation()
        self.check.side_effect = None
        self.robot.move_linear_pose.side_effect = ValueError('Motion failed')
        try:
            self.run_pick()
        except pickup.PickupPreflightError:
            self.fail('Motion errors must not be treated as recoverable preflight failures')
        except ValueError:
            pass
        else:
            self.fail('Expected motion failure to propagate')

    def test_explicit_reference_refresh_keeps_tool_poses_and_check_error_guard(self):
        data = sample_data()
        data.update(purpose='dice_pick_place', drop=[.3,.1,.22,0,1.57,0], tool='GRIPPER_1')
        workspace = detector.detect_workspace(workspace_frame())
        data['board_corners'] = (workspace.corners + [5, 0]).tolist()
        points = [[150,120],[490,120],[490,380],[150,380],[320,250]]
        with self.assertRaises(ValueError):
            pickup.replace_pick_pixels(data, points, workspace)
        updated = pickup.replace_pick_pixels(data, points, workspace, refresh_reference=True)
        self.assertEqual(updated['board_corners'], workspace.corners.tolist())
        self.assertEqual(updated['poses'], data['poses'])
        self.assertEqual(updated['overview'], data['overview'])
        # Changing a reference does not waive independent robot-coordinate validation.
        updated['poses'][4][0] += .1
        with self.assertRaisesRegex(ValueError, 'Independent check error'):
            PickCalibration(updated)

    def test_reference_refresh_cancel_keeps_draft_and_sends_no_commands(self):
        data = sample_data()
        workspace = detector.detect_workspace(workspace_frame())
        data['board_corners'] = (workspace.corners + [5, 0]).tolist()
        with (patch.object(pickup, 'read_frame', return_value=workspace_frame()),
              patch('builtins.input', return_value='cancel'),
              patch.object(pickup, 'save_pick_draft') as save,
              patch.object(pickup.cv2, 'namedWindow') as window,
              contextlib.redirect_stdout(io.StringIO())):
            result = pickup.reselect_pick_pixels(self.robot, None, data, refresh_reference=True)
        self.assertIs(result, data)
        save.assert_not_called()
        window.assert_not_called()
        self.assert_no_actuation()

    def test_reference_refresh_still_requires_original_overview(self):
        self.pose[0] += .02
        with (patch.object(pickup, 'read_frame') as camera,
              patch('builtins.input') as confirm):
            with self.assertRaises(ValueError):
                pickup.reselect_pick_pixels(self.robot, None, sample_data(), refresh_reference=True)
        camera.assert_not_called()
        confirm.assert_not_called()
        self.assert_no_actuation()

    def test_new_camera_overview_does_not_change_taught_grip_orientation(self):
        data = sample_data()
        grip = list(data['overview'])
        data.update(purpose='dice_pick_place', drop=[.3,.1,.22,0,1.57,0],
                    tool='GRIPPER_1', board_corners=[[100,80],[540,80],[540,420],[100,420]],
                    pickup_reference_pose=grip)
        data['overview'] = list(grip)
        data['overview'][5] += .04
        result = PickCalibration(data)
        target = result.target((300,250),(640,480))
        np.testing.assert_allclose(target[3:], grip[3:])
        np.testing.assert_allclose(result.overview, data['overview'])
        self.assertFalse(same_orientation(result.overview, target))
        # Original grip checks remain strict even when the camera reference changes.
        data['poses'][0][5] += .1
        with self.assertRaisesRegex(ValueError, 'Target 1'):
            PickCalibration(data)

    def test_refresh_at_different_orientation_can_be_cancelled_without_writes(self):
        data = sample_data()
        workspace = detector.detect_workspace(workspace_frame())
        data['board_corners'] = workspace.corners.tolist()
        self.pose[5] += .04
        with (patch.object(pickup, 'read_frame', return_value=workspace_frame()),
              patch('builtins.input', return_value='cancel') as confirm,
              patch.object(pickup, 'save_pick_draft') as save,
              contextlib.redirect_stdout(io.StringIO())):
            result = pickup.reselect_pick_pixels(self.robot, None, data, refresh_reference=True)
        confirm.assert_called_once()
        save.assert_not_called()
        self.assertIs(result, data)
        self.assert_no_actuation()

    def test_refresh_records_actual_overview_and_preserves_grip_reference(self):
        data = sample_data()
        workspace = detector.detect_workspace(workspace_frame())
        data['board_corners'] = workspace.corners.tolist()
        original = json.dumps(data)
        self.pose[5] += .04
        points = [[150,120],[490,120],[490,380],[150,380],[320,250]]
        def select(_window, callback):
            for x, y in points:
                callback(pickup.cv2.EVENT_LBUTTONDOWN, x, y, None, None)
        with (patch.object(pickup, 'read_frame', return_value=workspace_frame()),
              patch('builtins.input', return_value='SETUP UNCHANGED'),
              patch.object(pickup.cv2, 'namedWindow'),
              patch.object(pickup.cv2, 'setMouseCallback', side_effect=select),
              patch.object(pickup.cv2, 'imshow'),
              patch.object(pickup.cv2, 'waitKey', return_value=13),
              patch.object(pickup.cv2, 'destroyWindow'),
              patch.object(pickup, 'save_pick_draft') as save,
              contextlib.redirect_stdout(io.StringIO())):
            updated = pickup.reselect_pick_pixels(self.robot, None, data, refresh_reference=True)
        self.assertEqual(updated['overview'], self.pose.tolist())
        self.assertEqual(updated['pickup_reference_pose'], data['overview'])
        self.assertEqual(updated['poses'], data['poses'])
        self.assertEqual(updated['pixels'], points)
        self.assertEqual(json.dumps(data), original)
        save.assert_called_once_with(updated)
        self.assert_no_actuation()

    def test_fixed_target_uses_exact_taught_pose_without_interpolation(self):
        data = sample_data()
        data.update(purpose='dice_pick_place', drop=[.3,.1,.22,0,1.57,0],
                    tool='GRIPPER_1', board_corners=[[100,80],[540,80],[540,420],[100,420]])
        # A mapping error cannot alter the direct target's physically taught pose.
        data['poses'][0][0] += .02
        result = pickup.TaughtTargetPickup(data, 5)
        actual = result.target(data['pixels'][4], data['image_size'])
        np.testing.assert_allclose(actual, data['poses'][4])
        actual[0] = 999
        self.assertNotEqual(result.pickup[0], 999)
        for pixel in (data['pixels'][0], [350,250], [float('nan'),250]):
            with self.assertRaises(ValueError):
                result.target(pixel, data['image_size'])
        with self.assertRaises(ValueError):
            result.target(data['pixels'][4], [320,240])

    def test_fixed_target_still_rejects_wrong_height_or_grip_orientation(self):
        data = sample_data()
        data.update(purpose='dice_pick_place', drop=[.3,.1,.22,0,1.57,0],
                    tool='GRIPPER_1', board_corners=[[100,80],[540,80],[540,420],[100,420]])
        data['poses'][4][5] += .1
        with self.assertRaises(ValueError):
            pickup.TaughtTargetPickup(data, 5)
        data['poses'][4][5] -= .1
        data['drop'][2] = .5
        with self.assertRaises(ValueError):
            pickup.TaughtTargetPickup(data, 5)

    def test_hardware_tool_error_prevents_actuation(self):
        self.robot.get_hardware_status.return_value.hardware_errors[-1] = 1
        with self.assertRaisesRegex(pickup.PickupPreflightError, 'Tool: code 1'):
            self.run_pick()
        self.assert_no_actuation()

    def test_hardware_error_after_open_prevents_arm_motion(self):
        def opening_error(**_kwargs):
            self.robot.get_hardware_status.return_value.hardware_errors[-1] = 1
        self.robot.open_gripper.side_effect = opening_error
        with self.assertRaisesRegex(ValueError, 'Tool: code 1'):
            self.run_pick()
        self.robot.open_gripper.assert_called_once()
        self.robot.move_linear_pose.assert_not_called()
        self.robot.close_gripper.assert_not_called()

    def test_delayed_hardware_fault_after_approach_prevents_descent(self):
        def fault_during_approach(pose):
            self.move(pose)
            self.robot.get_hardware_status.return_value.hardware_errors[-1] = 1
        self.robot.move_linear_pose.side_effect = fault_during_approach
        with self.assertRaisesRegex(ValueError, 'Tool: code 1'):
            self.run_pick()
        self.robot.move_linear_pose.assert_called_once()
        self.robot.close_gripper.assert_not_called()
        self.robot.open_gripper.assert_called_once_with(**pickup.GRIPPER_OPTIONS)

    def test_upper_moves_have_separate_tolerance_from_pickup_and_drop(self):
        with patch.object(pickup, 'wait_for_pose') as wait:
            self.assertTrue(self.run_pick())
        self.assertEqual(wait.call_count, 7)
        # Approach, lift, transfer, lift and camera return are upper movements.
        for index in (0, 2, 3, 5, 6):
            self.assertEqual(wait.call_args_list[index].kwargs,
                             {'angle_tolerance': pickup.OVERVIEW_ANGLE_TOLERANCE})
        # Never relax the check immediately before closing/opening at the work surface.
        for index in (1, 4):
            self.assertEqual(wait.call_args_list[index].kwargs, {})

    def test_misplaced_fixed_target_reports_distance_without_changing_teaching(self):
        data = sample_data()
        data.update(purpose='dice_pick_place', drop=[.3,.1,.22,0,1.57,0],
                    tool='GRIPPER_1', board_corners=[[100,80],[540,80],[540,420],[100,420]])
        target = pickup.TaughtTargetPickup(data, 5)
        original = json.dumps(data)
        with self.assertRaisesRegex(ValueError, '12.2 px'):
            target.target((307,240), (640,480))
        self.assertEqual(json.dumps(data), original)
        np.testing.assert_allclose(target.target((300,250),(640,480)), data['poses'][4])

    def test_fixed_view_refresh_only_updates_in_memory_markers(self):
        data = sample_data()
        data.update(purpose='dice_pick_place', drop=[.3,.1,.22,0,1.57,0],
                    tool='GRIPPER_1', board_corners=[[104,80],[544,80],[544,420],[104,420]])
        original = json.dumps(data)
        target = pickup.TaughtTargetPickup(data, 5)
        with (patch.object(pickup, 'require_live_camera') as live,
              patch.object(pickup, 'read_frame', return_value=workspace_frame()),
              patch.object(pickup, 'save_pick_draft') as save,
              contextlib.redirect_stdout(io.StringIO())):
            self.assertTrue(pickup.refresh_fixed_reference(
                self.robot, target, target.intrinsics, 'robot', lambda _: 'FIXED SETUP'))
        live.assert_called_once_with('robot')
        save.assert_not_called()
        np.testing.assert_allclose(target.board, [[100,80],[540,80],[540,420],[100,420]])
        np.testing.assert_allclose(target.pickup, data['poses'][4])
        self.assertEqual(json.dumps(data), original)
        with self.assertRaises(ValueError):
            target.target((310,250),(640,480))
        self.assert_no_actuation()

    def test_fixed_view_refresh_cancel_keeps_reference(self):
        data = sample_data()
        data.update(purpose='dice_pick_place', drop=[.3,.1,.22,0,1.57,0],
                    tool='GRIPPER_1', board_corners=[[104,80],[544,80],[544,420],[104,420]])
        target = pickup.TaughtTargetPickup(data, 5)
        with (patch.object(pickup, 'read_frame') as camera,
              contextlib.redirect_stdout(io.StringIO())):
            self.assertFalse(pickup.refresh_fixed_reference(
                self.robot, target, target.intrinsics, 'robot', lambda _: 'cancel'))
        camera.assert_not_called()
        np.testing.assert_allclose(target.board, data['board_corners'])
        self.assert_no_actuation()

    def test_full_cycle(self):
        self.assertTrue(self.run_pick())
        self.check.assert_called_once()
        for call in self.robot.open_gripper.call_args_list:
            self.assertEqual(call.kwargs, pickup.GRIPPER_OPTIONS)
        self.robot.close_gripper.assert_called_once_with(**pickup.GRIPPER_OPTIONS)
        names = [c[0] for c in self.robot.mock_calls
                 if c[0] in ('open_gripper', 'close_gripper', 'move_linear_pose')]
        self.assertEqual(names, ['open_gripper', 'move_linear_pose', 'move_linear_pose',
                                'close_gripper', 'move_linear_pose', 'move_linear_pose',
                                'move_linear_pose', 'open_gripper', 'move_linear_pose', 'move_linear_pose'])
        poses = [c.args[0] for c in self.robot.move_linear_pose.call_args_list]
        np.testing.assert_allclose(poses[1], self.mapping.target((300, 250), (640, 480)))
        np.testing.assert_allclose(poses[4], self.mapping.drop)
        np.testing.assert_allclose(self.pose, self.mapping.overview)

    def test_cancel(self):
        self.assertFalse(self.run_pick('cancel'))
        self.assert_no_actuation()

    def test_scene_changed(self):
        self.check.side_effect = ValueError('Board shifted')
        with self.assertRaisesRegex(ValueError, 'shifted'):
            self.run_pick()
        self.assert_no_actuation()

    def test_learning_mode(self):
        self.robot.get_learning_mode.return_value = True
        with self.assertRaisesRegex(ValueError, 'learning'):
            self.run_pick()
        self.assert_no_actuation()

    def test_wrong_tool(self):
        self.robot.get_current_tool_id.return_value = ToolID.VACUUM_PUMP_1
        with self.assertRaisesRegex(ValueError, 'gripper'):
            self.run_pick()
        self.assert_no_actuation()

    def test_outside_calibration(self):
        self.mapping.boundary += 500
        with self.assertRaisesRegex(ValueError, 'outside'):
            self.run_pick()
        self.assert_no_actuation()

    def test_lift_error_stops_without_releasing(self):
        def fail_on_lift(pose):
            if self.robot.close_gripper.called:
                raise RuntimeError('Lift failed')
            self.move(pose)
        self.robot.move_linear_pose.side_effect = fail_on_lift
        with self.assertRaisesRegex(RuntimeError, 'Lift failed'):
            self.run_pick()
        self.assertEqual(self.robot.move_linear_pose.call_count, 3)
        self.assertEqual(self.robot.open_gripper.call_count, 1)

    def test_board_change_or_loss(self):
        workspace = detector.detect_workspace(workspace_frame())
        self.mapping.check_board(workspace)
        workspace.corners += 10
        for value in (workspace, None):
            with self.assertRaisesRegex(ValueError, 'Board'):
                self.mapping.check_board(value)

    def test_camera_calibration_rejected(self):
        with self.assertRaisesRegex(ValueError, 'separate'):
            PickCalibration(sample_data())

    def test_perspective_side_midpoints(self):
        workspace = detector.detect_workspace(workspace_frame([(130, 70), (500, 100), (550, 410), (80, 380)]))
        for point, relative in zip(workspace.side_points, [(0.5, 0), (1, 0.5), (0.5, 1), (0, 0.5)]):
            np.testing.assert_allclose(workspace.relative_position(point), relative, atol=1e-6)


if __name__ == '__main__':
    unittest.main()
