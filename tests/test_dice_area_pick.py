import contextlib
import copy
import io
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import cv2
import numpy as np

from test_dice_camera_calibration import detector, sample_data, workspace_frame
from test_dice_pick_place import pickup


def area_data():
    data = sample_data()
    data.update(purpose='dice_pick_place', drop=[.3, .1, .22, 0, 1.57, 0],
                board_corners=[[100, 80], [540, 80], [540, 420], [100, 420]], tool='GRIPPER_1')
    return data


class AreaMappingTests(unittest.TestCase):
    def setUp(self):
        self.data = area_data()
        self.mapping = pickup.PickCalibration(self.data)
        self.workspace = detector.detect_workspace(workspace_frame())

    def test_many_interior_points_have_distinct_analytic_robot_targets(self):
        for x in np.linspace(120, 480, 7):
            for y in np.linspace(120, 380, 6):
                with self.subTest(x=x, y=y):
                    pose = pickup.resolve_pick_target(self.mapping, (x, y), (640, 480), self.workspace)
                    np.testing.assert_allclose(pose[:3], [.1+x*.0005, -.2+y*.0005, .2], atol=1e-6)
                    np.testing.assert_allclose(pose[3:], self.data['overview'][3:])

    def test_marker_registration_preserves_target_under_small_perspective_change(self):
        transform = np.array([[1, 0, 1], [0, 1, -1], [.000002, 0, 1]], dtype=np.float64)
        current = copy.deepcopy(self.workspace)
        current.corners = cv2.perspectiveTransform(
            self.workspace.corners.astype(np.float32).reshape(1, 4, 2), transform)[0]
        original = np.array([[[240, 190]]], dtype=np.float32)
        current_pixel = cv2.perspectiveTransform(original, transform)[0, 0]
        target = pickup.resolve_pick_target(self.mapping, current_pixel, (640, 480), current)
        np.testing.assert_allclose(target[:3], [.22, -.105, .2], atol=1e-6)
        expected_boundary = cv2.perspectiveTransform(self.mapping.boundary.reshape(1, 4, 2), transform)[0]
        np.testing.assert_allclose(self.mapping.boundary_in_workspace(current), expected_boundary, atol=1e-4)

    def test_no_extrapolation_outside_taught_area_even_inside_board(self):
        self.assertNotEqual(self.workspace.mask[250, 520], 0)
        with self.assertRaisesRegex(ValueError, 'outside'):
            pickup.resolve_pick_target(self.mapping, (520, 250), (640, 480), self.workspace)

    def test_missing_shifted_markers_resolution_and_invalid_pixels_are_blocked(self):
        shifted = copy.deepcopy(self.workspace)
        shifted.corners += [10, 0]
        with self.assertRaises(ValueError):
            self.mapping.target_in_workspace((300, 250), (640, 480), shifted)
        with self.assertRaises(ValueError):
            self.mapping.target_in_workspace((300, 250), (640, 480), None)
        with self.assertRaises(ValueError):
            self.mapping.target_in_workspace((300, 250), (320, 240), self.workspace)
        for point in ([float('nan'), 250], [1, 2, 3]):
            with self.assertRaises(ValueError):
                self.mapping.target_in_workspace(point, (640, 480), self.workspace)

    def test_diagnostic_outline_does_not_enable_motion_with_shifted_reference(self):
        moved = copy.deepcopy(self.workspace)
        moved.corners += [4, 0]
        boundary = self.mapping.boundary_in_workspace(moved, validate_reference=False)
        np.testing.assert_allclose(boundary, self.mapping.boundary + [4,0], atol=1e-4)
        with self.assertRaises(ValueError):
            self.mapping.boundary_in_workspace(moved)
        with self.assertRaises(ValueError):
            pickup.resolve_pick_target(self.mapping, (244,190), (640,480), moved)

    def test_explicit_board_area_accepts_inside_board_outside_old_taught_hull(self):
        with self.assertRaises(ValueError):
            self.mapping.target_in_workspace((520,250), (640,480), self.workspace)
        original_matrix = self.mapping.matrix.copy()
        original_check_error = self.mapping.error
        self.mapping.enable_board_area()
        pose = self.mapping.target_in_workspace((520,250), (640,480), self.workspace)
        np.testing.assert_allclose(pose[:3], [.36,-.075,.2], atol=1e-6)
        np.testing.assert_array_equal(self.mapping.boundary, self.mapping.board)
        np.testing.assert_array_equal(self.mapping.matrix, original_matrix)
        self.assertEqual(self.mapping.error, original_check_error)
        with self.assertRaises(ValueError):
            self.mapping.target_in_workspace((550,250), (640,480), self.workspace)

    def test_board_extension_rejects_projection_singularity(self):
        self.mapping.matrix[2] = [1,0,-520]
        with self.assertRaisesRegex(ValueError, 'singularity'):
            self.mapping.enable_board_area()

    def test_bad_fifth_point_is_not_hidden_by_fitting_all_five(self):
        self.data['poses'][4][0] += .009
        with self.assertRaisesRegex(ValueError, 'Independent check error'):
            pickup.PickCalibration(self.data)

    def test_fixed_target_mode_still_requires_its_original_location(self):
        target = pickup.TaughtTargetPickup(self.data, 5)
        with self.assertRaisesRegex(ValueError, 'Fixed target'):
            pickup.resolve_pick_target(target, (240, 190), (640, 480), self.workspace)


class AreaMotionTests(unittest.TestCase):
    def setUp(self):
        self.mapping = pickup.PickCalibration(area_data())
        self.workspace = detector.detect_workspace(workspace_frame())
        self.workspace.corners += [2, 0]
        self.pose = self.mapping.overview.copy()
        self.robot = Mock()
        self.robot.get_pose.return_value.to_list.side_effect = lambda: self.pose.tolist()
        self.robot.get_current_tool_id.return_value.name = 'GRIPPER_1'
        self.robot.get_learning_mode.return_value = False
        self.robot.get_hardware_status.return_value = SimpleNamespace(
            connection_up=True, calibration_needed=False, calibration_in_progress=False,
            motor_names=['Tool'], hardware_errors=[0], error_message='')
        self.robot.move_linear_pose.side_effect = self.move
        self.revalidate = Mock()

    def move(self, pose):
        self.pose = np.asarray(pose)

    def call(self, function, answer):
        with patch('builtins.input', return_value=answer), contextlib.redirect_stdout(io.StringIO()):
            return function(self.robot, self.mapping, (242, 190), (640, 480),
                            self.revalidate, workspace=self.workspace)

    def test_preview_uses_selected_xy_and_never_descends_or_actuates_gripper(self):
        self.assertTrue(self.call(pickup.preview_pick, 'PREVIEW'))
        self.revalidate.assert_called_once()
        moves = [call.args[0] for call in self.robot.move_linear_pose.call_args_list]
        self.assertEqual(len(moves), 2)
        np.testing.assert_allclose(moves[0][:3], [.22, -.105, .4], atol=1e-6)
        np.testing.assert_allclose(moves[1], self.mapping.overview)
        self.robot.open_gripper.assert_not_called()
        self.robot.close_gripper.assert_not_called()
        self.robot.tool_reboot.assert_not_called()

    def test_full_cycle_descends_to_new_selected_xy_not_point_five(self):
        self.assertTrue(self.call(pickup.pick_and_place, 'PICK'))
        moves = [call.args[0] for call in self.robot.move_linear_pose.call_args_list]
        self.assertEqual(len(moves), 7)
        np.testing.assert_allclose(moves[1][:3], [.22, -.105, .2], atol=1e-6)
        self.robot.close_gripper.assert_called_once()
        np.testing.assert_allclose(moves[4], self.mapping.drop)

    def test_area_view_refresh_keeps_canonical_mapping_and_allows_new_cube_positions(self):
        original_board = self.mapping.board.copy()
        original_pose = self.mapping.poses[4].copy()
        shifted = copy.deepcopy(self.workspace)
        shifted.corners = self.mapping.board + [4, 0]
        with (patch.object(pickup, 'require_live_camera'),
              patch.object(pickup, 'read_frame', return_value=workspace_frame()),
              patch.object(pickup, 'detect_workspace', return_value=shifted),
              patch.object(pickup, 'save_pick_draft') as save,
              contextlib.redirect_stdout(io.StringIO())):
            self.assertTrue(pickup.refresh_area_reference(
                self.robot, self.mapping, self.mapping.intrinsics, 'robot', lambda _: 'BOARD FIXED'))
        np.testing.assert_array_equal(self.mapping.board, original_board)
        np.testing.assert_array_equal(self.mapping.poses[4], original_pose)
        np.testing.assert_array_equal(self.mapping.view_board, shifted.corners)
        save.assert_not_called()
        # A new arbitrary position in the refreshed frame still maps to the ORIGINAL robot XY.
        pose = pickup.resolve_pick_target(self.mapping, (244,190), (640,480), shifted)
        np.testing.assert_allclose(pose[:3], [.22,-.105,.2], atol=1e-6)
        self.robot.move_linear_pose.assert_not_called()
        self.robot.open_gripper.assert_not_called()
        self.robot.close_gripper.assert_not_called()

    def test_explicit_area_capture_accepts_new_camera_orientation_not_new_grip(self):
        original_grip = self.mapping.pickup_reference.copy()
        self.pose[5] += np.deg2rad(3.51)
        with (patch.object(pickup,'require_live_camera'),
              patch.object(pickup,'read_frame',return_value=workspace_frame()),
              contextlib.redirect_stdout(io.StringIO())):
            self.assertTrue(pickup.refresh_area_reference(
                self.robot,self.mapping,self.mapping.intrinsics,'robot',lambda _: 'BOARD FIXED'))
        np.testing.assert_allclose(self.mapping.overview,self.pose)
        np.testing.assert_allclose(self.mapping.pickup_reference,original_grip)
        target=self.mapping.target((240,190),(640,480))
        np.testing.assert_allclose(target[3:],original_grip[3:])
        self.robot.move_linear_pose.assert_not_called()
        self.robot.close_gripper.assert_not_called()

    def test_area_capture_still_rejects_large_rotation_or_position_change(self):
        for component, delta in [(5,np.deg2rad(15)),(0,.003)]:
            self.pose=self.mapping.overview.copy()
            self.pose[component]+=delta
            confirm=Mock()
            with self.assertRaises(ValueError):
                pickup.refresh_area_reference(self.robot,self.mapping,self.mapping.intrinsics,'robot',confirm)
            confirm.assert_not_called()
        self.robot.move_linear_pose.assert_not_called()

    def test_area_view_refresh_cancel_changes_nothing(self):
        old_board = self.mapping.view_board.copy()
        with (patch.object(pickup, 'read_frame') as camera,
              contextlib.redirect_stdout(io.StringIO())):
            self.assertFalse(pickup.refresh_area_reference(
                self.robot, self.mapping, self.mapping.intrinsics, 'robot', lambda _: 'cancel'))
        camera.assert_not_called()
        np.testing.assert_array_equal(self.mapping.view_board, old_board)

    def test_area_view_refresh_never_proceeds_on_motor_fault(self):
        self.robot.get_hardware_status.return_value.connection_up = False
        confirm = Mock()
        with self.assertRaises(ValueError):
            pickup.refresh_area_reference(self.robot, self.mapping, self.mapping.intrinsics, 'robot', confirm)
        confirm.assert_not_called()
        self.robot.move_linear_pose.assert_not_called()

    def test_cancel_preview_does_not_actuate(self):
        self.assertFalse(self.call(pickup.preview_pick, 'cancel'))
        self.robot.move_linear_pose.assert_not_called()
        self.robot.set_arm_max_velocity.assert_not_called()
        self.robot.open_gripper.assert_not_called()
        self.robot.close_gripper.assert_not_called()

    def test_hardware_fault_before_preview_blocks_every_actuator(self):
        self.robot.get_hardware_status.return_value.hardware_errors = [1]
        with self.assertRaises(pickup.PickupPreflightError):
            self.call(pickup.preview_pick, 'PREVIEW')
        self.robot.move_linear_pose.assert_not_called()
        self.robot.set_arm_max_velocity.assert_not_called()

    def test_scene_change_during_confirmation_blocks_preview(self):
        self.revalidate.side_effect = ValueError('Die moved')
        with self.assertRaises(pickup.PickupPreflightError):
            self.call(pickup.preview_pick, 'PREVIEW')
        self.robot.move_linear_pose.assert_not_called()

    def test_wait_rechecks_pose_without_resending_motion(self):
        with (patch.object(pickup,'wait_for_pose',side_effect=[ValueError('Delayed angle'),None]) as wait,
              patch('builtins.input',return_value='WAIT'),contextlib.redirect_stdout(io.StringIO())):
            pickup.wait_for_preview_pose(self.robot,self.mapping.overview)
        self.assertEqual(wait.call_count,2)
        self.robot.move_linear_pose.assert_not_called()
        self.robot.close_gripper.assert_not_called()

    def test_wait_is_not_offered_on_a_hardware_fault(self):
        self.robot.get_hardware_status.return_value.connection_up=False
        with (patch.object(pickup,'wait_for_pose',side_effect=ValueError('Not reached')),
              patch('builtins.input') as confirm):
            with self.assertRaises(ValueError):
                pickup.wait_for_preview_pose(self.robot,self.mapping.overview)
        confirm.assert_not_called()
        self.robot.move_linear_pose.assert_not_called()

    def test_motion_failure_stops_without_automatic_return(self):
        self.robot.move_linear_pose.side_effect = RuntimeError('Trajectory aborted')
        with self.assertRaisesRegex(RuntimeError, 'Trajectory aborted'):
            self.call(pickup.preview_pick, 'PREVIEW')
        self.robot.move_linear_pose.assert_called_once()
        self.robot.open_gripper.assert_not_called()
        self.robot.close_gripper.assert_not_called()


class AreaDetectorIntegrationTests(unittest.TestCase):
    def test_v_key_refreshes_view_without_executing_pick_or_preview(self):
        robot = Mock()
        mapping = detector.PickCalibration(area_data())
        with (
            patch.object(sys,'argv',['detector','--enable-motion','--preview-pick']),
            patch.object(detector,'NiryoRobot',return_value=robot),
            patch.object(detector,'PICK_FILE') as file,
            patch.object(detector,'load_pick_calibration',return_value=mapping),
            patch.object(detector,'load_calibration',return_value=Mock()),
            patch.object(detector,'read_frame',return_value=workspace_frame()),
            patch.object(detector,'refresh_area_reference',return_value=True) as refresh,
            patch.object(detector,'preview_pick') as preview,
            patch.object(detector,'pick_and_place') as pick,
            patch.object(detector.cv2,'namedWindow'),
            patch.object(detector.cv2,'setMouseCallback'),
            patch.object(detector.cv2,'imshow'),
            patch.object(detector.cv2,'destroyAllWindows'),
            patch.object(detector.cv2,'waitKey',side_effect=[ord('v'),ord('q')]),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            file.exists.return_value=True
            detector.main()
        refresh.assert_called_once()
        self.assertIs(refresh.call_args.args[1],mapping)
        preview.assert_not_called()
        pick.assert_not_called()
        robot.move_linear_pose.assert_not_called()
        robot.close_gripper.assert_not_called()

    def test_failed_session_refresh_never_falls_back_to_motion_with_old_reference(self):
        robot = Mock()
        mapping = detector.PickCalibration(area_data())
        with (
            patch.object(sys, 'argv', ['detector', '--enable-motion', '--preview-pick', '--refresh-area-view']),
            patch.object(detector, 'NiryoRobot', return_value=robot),
            patch.object(detector, 'PICK_FILE') as file,
            patch.object(detector, 'load_pick_calibration', return_value=mapping),
            patch.object(detector, 'refresh_area_reference', side_effect=ValueError('Hardware disconnected')),
            patch.object(detector, 'preview_pick') as preview,
            patch.object(detector, 'pick_and_place') as full_pick,
            patch.object(detector, 'read_frame') as frame,
            patch.object(detector.cv2, 'destroyAllWindows'),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            file.exists.return_value = True
            detector.main()
        preview.assert_not_called()
        full_pick.assert_not_called()
        frame.assert_not_called()
        robot.close_connection.assert_called_once()

    def test_general_preview_selects_a_die_away_from_taught_point_five(self):
        frame = workspace_frame()
        cv2.rectangle(frame, (220, 170), (260, 210), (10, 10, 10), -1)
        mapping = detector.PickCalibration(area_data())
        robot = Mock()
        robot.get_pose.return_value.to_list.return_value = mapping.overview.tolist()
        def click(_name, callback):
            callback(cv2.EVENT_LBUTTONDOWN, 240, 190, 0, None)
        with (
            patch.object(sys, 'argv', ['detector', '--enable-motion', '--preview-pick']),
            patch.object(detector, 'NiryoRobot', return_value=robot),
            patch.object(detector, 'PICK_FILE') as file,
            patch.object(detector, 'load_pick_calibration', return_value=mapping),
            patch.object(detector, 'load_calibration', return_value=Mock()),
            patch.object(detector, 'read_frame', return_value=frame),
            patch.object(detector, 'require_live_camera'),
            patch.object(detector.cv2, 'namedWindow'),
            patch.object(detector.cv2, 'setMouseCallback', side_effect=click),
            patch.object(detector.cv2, 'imshow'),
            patch.object(detector.cv2, 'destroyAllWindows'),
            patch.object(detector.cv2, 'waitKey', side_effect=[ord('p'), ord('q')]),
            patch.object(detector, 'preview_pick', return_value=True) as preview,
            patch.object(detector, 'pick_and_place') as full_pick,
            contextlib.redirect_stdout(io.StringIO()),
        ):
            file.exists.return_value = True
            detector.main()
        preview.assert_called_once()
        self.assertEqual(preview.call_args.args[2], (240, 190))
        self.assertIsNotNone(preview.call_args.kwargs['workspace'])
        full_pick.assert_not_called()
        robot.close_connection.assert_called_once()


if __name__ == '__main__':
    unittest.main()
