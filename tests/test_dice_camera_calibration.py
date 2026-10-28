import copy
import importlib.util
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "experiments" / "vision" / "dice_camera_calibration.py"
)
SPEC = importlib.util.spec_from_file_location("dice_camera_calibration", MODULE_PATH)
calibration = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(calibration)
DETECTOR_SPEC = importlib.util.spec_from_file_location(
    "rectified_detector", MODULE_PATH.with_name("dice_object_detection-rectified.py")
)
detector = importlib.util.module_from_spec(DETECTOR_SPEC)
with patch.dict(sys.modules, {"dice_camera_calibration": calibration}):
    DETECTOR_SPEC.loader.exec_module(detector)


def sample_data():
    pixels = [[100, 100], [500, 100], [500, 400], [100, 400], [300, 250]]
    return {
        "version": 1,
        "image_size": [640, 480],
        "overview": [0.25, 0, 0.4, 0, 1.57, 0],
        "pixels": pixels,
        "poses": [
            [0.1 + x * 0.0005, -0.2 + y * 0.0005, 0.2, 0, 1.57, 0]
            for x, y in pixels
        ],
        "camera_matrix": np.eye(3).tolist(),
        "distortion": [[0, 0, 0, 0, 0]],
    }


def workspace_frame(corners=None):
    frame = np.full((480, 640, 3), 240, dtype=np.uint8)
    if corners is None:
        corners = [(100, 80), (540, 80), (540, 420), (100, 420)]
    for point in corners:
        for radius, color in [(16, 10), (12, 240), (8, 10), (4, 240)]:
            cv2.circle(frame, point, radius, (color, color, color), -1)
    return frame


class FakeRobot:
    def __init__(self, pose, learning=False):
        self.pose = pose
        self.learning = learning
        self.moves = []
        self.speeds = []

    def get_pose(self):
        return self

    def to_list(self):
        return list(self.pose)

    def get_learning_mode(self):
        return self.learning

    def set_arm_max_velocity(self, speed):
        self.speeds.append(speed)

    def move_linear_pose(self, pose):
        self.moves.append(pose)
        self.pose = pose


class CalibrationTests(unittest.TestCase):
    def setUp(self):
        self.data = sample_data()
        self.mapping = calibration.CameraCalibration(self.data)

    def test_known_and_interpolated_targets(self):
        for point, expected in [
            ([300, 250], [0.25, -0.075, 0.2]),
            ([200, 200], [0.20, -0.1, 0.2]),
            ([100, 100], [0.15, -0.15, 0.2]),
        ]:
            np.testing.assert_allclose(
                self.mapping.target(point, (640, 480))[:3], expected, atol=1e-6
            )
        self.assertLess(self.mapping.error, 1e-6)

    def test_reject_outside_and_changed_resolution(self):
        with self.assertRaisesRegex(ValueError, "outside"):
            self.mapping.target([50, 200], (640, 480))
        with self.assertRaisesRegex(ValueError, "resolution"):
            self.mapping.target([300, 250], (1280, 960))
        with self.assertRaisesRegex(ValueError, "finite"):
            self.mapping.target([np.nan, 250], (640, 480))

    def test_independent_validation_rejects_inaccurate_mapping(self):
        self.data["poses"][4][0] += 0.006
        with self.assertRaisesRegex(ValueError, "exceeds 5 mm"):
            calibration.CameraCalibration(self.data)

    def test_reject_inconsistent_height_orientation_and_overview(self):
        for index, coordinate, change, message in [
            (2, 2, 0.01, "SAME safe close-up"),
            (2, 3, 0.1, "SAME camera orientation"),
        ]:
            data = copy.deepcopy(self.data)
            data["poses"][index][coordinate] += change
            with self.assertRaisesRegex(ValueError, message):
                calibration.CameraCalibration(data)
        self.data["overview"][2] = 0.19
        with self.assertRaisesRegex(ValueError, "above"):
            calibration.CameraCalibration(self.data)

    def test_reject_degenerate_points(self):
        self.data["pixels"][3] = self.data["pixels"][2]
        with self.assertRaisesRegex(ValueError, "convex"):
            calibration.CameraCalibration(self.data)

    def test_reject_invalid_check_point(self):
        self.data["pixels"][4] = [20, 20]
        with self.assertRaisesRegex(ValueError, "inside the first four"):
            calibration.CameraCalibration(self.data)

    def test_pose_guard_and_wrapped_angles(self):
        calibration.require_pose(self.data["overview"], self.data["overview"])
        wrapped = list(self.data["overview"])
        wrapped[5] += 2 * np.pi
        calibration.require_pose(wrapped, self.data["overview"])
        moved = list(self.data["overview"])
        moved[0] += 0.01
        with self.assertRaisesRegex(ValueError, "required taught pose"):
            calibration.require_pose(moved, self.data["overview"])

    def test_no_confirmation_means_no_motion(self):
        robot = FakeRobot(self.data["overview"])
        with patch("builtins.input", return_value="cancel"):
            self.assertFalse(calibration.confirmed_move(
                robot, self.mapping.overview, [self.mapping.overview]
            ))
        self.assertEqual(robot.moves, [])
        self.assertEqual(robot.speeds, [])

    def test_confirmed_waypoints_and_speed(self):
        robot = FakeRobot(self.data["overview"])
        target = self.mapping.target([300, 250], (640, 480))
        raised = target.copy()
        raised[2] = self.mapping.overview[2]
        with patch("builtins.input", return_value="MOVE"):
            self.assertTrue(calibration.confirmed_move(
                robot, self.mapping.overview, [raised, target]
            ))
        np.testing.assert_allclose(robot.moves, [raised, target])
        self.assertEqual(robot.speeds, [10])

    def test_robot_moved_during_confirmation(self):
        robot = FakeRobot(list(self.data["overview"]))

        def move_before_confirm(_prompt):
            robot.pose[0] += 0.01
            return "MOVE"

        with patch("builtins.input", side_effect=move_before_confirm):
            with self.assertRaisesRegex(ValueError, "required taught pose"):
                calibration.confirmed_move(
                    robot, self.mapping.overview, [self.mapping.overview]
                )
        self.assertEqual(robot.moves, [])

    def test_learning_mode_blocks_motion(self):
        robot = FakeRobot(self.data["overview"], learning=True)
        with patch("builtins.input", return_value="MOVE"):
            with self.assertRaisesRegex(ValueError, "learning mode"):
                calibration.confirmed_move(
                    robot, self.mapping.overview, [self.mapping.overview]
                )
        self.assertEqual(robot.moves, [])

    def test_motion_failure_stops_remaining_waypoints(self):
        robot = FakeRobot(self.data["overview"])
        target = self.mapping.target([300, 250], (640, 480))
        with (
            patch("builtins.input", return_value="MOVE"),
            patch.object(robot, "move_linear_pose", side_effect=RuntimeError("blocked")) as move,
        ):
            with self.assertRaisesRegex(RuntimeError, "blocked"):
                calibration.confirmed_move(robot, self.mapping.overview, [target, target])
        self.assertEqual(move.call_count, 1)

    def test_detector_preserves_rectified_outputs_in_automatic_workspace(self):
        frame = workspace_frame()
        for x, y in [(250, 100), (350, 180)]:
            cv2.rectangle(frame, (x, y), (x + 40, y + 40), (10, 10, 10), -1)
            cv2.circle(frame, (x + 20, y + 20), 4, (240, 240, 240), -1)
        cv2.rectangle(frame, (20, 20), (60, 60), (10, 10, 10), -1)
        detections, workspace = detector.detect_dice(frame)
        self.assertEqual(len(detections), 2)
        self.assertIsNotNone(workspace)
        np.testing.assert_allclose(
            workspace.corners, [(100, 80), (540, 80), (540, 420), (100, 420)], atol=1
        )
        for die in detections:
            self.assertEqual(die["warped"].shape, (256, 256, 3))
            self.assertEqual(die["warped_binary"].shape, (256, 256))

    def test_missing_calibration_blocks_motion_in_main(self):
        robot = Mock()
        frame = workspace_frame()
        with (
            patch.object(sys, "argv", ["rectified_detector", "--enable-motion"]),
            patch.object(detector, "NiryoRobot", return_value=robot),
            patch.object(detector, "CALIBRATION_FILE") as file,
            patch.object(detector, "read_frame", return_value=frame),
            patch.object(detector.cv2, "namedWindow"),
            patch.object(detector.cv2, "setMouseCallback"),
            patch.object(detector.cv2, "imshow"),
            patch.object(detector.cv2, "destroyAllWindows"),
            patch.object(detector.cv2, "waitKey", side_effect=[ord("m"), ord("q")]),
            patch.object(detector, "confirmed_move") as move,
        ):
            file.exists.return_value = False
            detector.main()
        move.assert_not_called()
        robot.move_linear_pose.assert_not_called()
        robot.close_connection.assert_called_once()

    def test_workspace_without_dice_excludes_ring_markers(self):
        detections, workspace = detector.detect_dice(workspace_frame())
        self.assertIsNotNone(workspace)
        self.assertEqual(detections, [])

    def test_normalized_board_positions(self):
        workspace = detector.detect_workspace(workspace_frame())
        self.assertIsNotNone(workspace)
        for pixel, expected in [
            ((100, 80), (0, 0)), ((540, 80), (1, 0)),
            ((540, 420), (1, 1)), ((100, 420), (0, 1)),
            ((320, 250), (0.5, 0.5)), ((210, 165), (0.25, 0.25)),
        ]:
            np.testing.assert_allclose(workspace.relative_position(pixel), expected, atol=1e-6)

    def test_positions_use_perspective_not_bounding_rectangle(self):
        corners = np.array([(130, 70), (500, 100), (550, 410), (80, 380)],
                           dtype=np.float32)
        relative = np.array([[[0.25, 0.75]]], dtype=np.float32)
        forward = cv2.getPerspectiveTransform(
            np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float32), corners
        )
        pixel = cv2.perspectiveTransform(relative, forward)[0, 0]
        workspace = detector.detect_workspace(workspace_frame(
            [tuple(map(int, point)) for point in corners]
        ))
        self.assertIsNotNone(workspace)
        np.testing.assert_allclose(
            workspace.relative_position(pixel), (0.25, 0.75), atol=1e-6
        )

    def test_detected_dice_include_board_position(self):
        frame = workspace_frame()
        cv2.rectangle(frame, (300, 230), (340, 270), (10, 10, 10), -1)
        dice, _ = detector.detect_dice(frame)
        self.assertEqual(len(dice), 1)
        np.testing.assert_allclose(dice[0]["workspace_position"], (0.5, 0.5), atol=1e-6)

    def test_default_mode_cannot_move_even_with_saved_calibration(self):
        robot = Mock()
        with (
            patch.object(sys, "argv", ["rectified_detector"]),
            patch.object(detector, "NiryoRobot", return_value=robot),
            patch.object(detector, "CALIBRATION_FILE") as file,
            patch.object(detector, "load_calibration") as load,
            patch.object(detector, "read_frame", return_value=workspace_frame()),
            patch.object(detector.cv2, "namedWindow"),
            patch.object(detector.cv2, "setMouseCallback"),
            patch.object(detector.cv2, "imshow"),
            patch.object(detector.cv2, "destroyAllWindows"),
            patch.object(detector.cv2, "waitKey",
                         side_effect=[ord("m"), ord("b"), ord("q")]),
            patch.object(detector, "confirmed_move") as move,
        ):
            file.exists.return_value = True
            detector.main()
        file.exists.assert_not_called()
        load.assert_not_called()
        move.assert_not_called()
        robot.get_pose.assert_not_called()
        robot.set_arm_max_velocity.assert_not_called()
        robot.move_linear_pose.assert_not_called()

    def test_click_reports_relative_position_without_motion(self):
        frame = workspace_frame()
        cv2.rectangle(frame, (300, 230), (340, 270), (10, 10, 10), -1)
        robot = Mock()

        def click_center(_window, callback):
            callback(cv2.EVENT_LBUTTONDOWN, 320, 250, 0, None)

        with (
            patch.object(sys, "argv", ["rectified_detector"]),
            patch.object(detector, "NiryoRobot", return_value=robot),
            patch.object(detector, "read_frame", return_value=frame),
            patch.object(detector.cv2, "namedWindow"),
            patch.object(detector.cv2, "setMouseCallback", side_effect=click_center),
            patch.object(detector.cv2, "imshow"),
            patch.object(detector.cv2, "destroyAllWindows"),
            patch.object(detector.cv2, "waitKey", return_value=ord("q")),
            patch("builtins.print") as output,
        ):
            detector.main()
        messages = [str(call.args[0]) for call in output.call_args_list if call.args]
        self.assertIn(
            "Die 1: pixel centre=(320, 250), board x=0.5000, y=0.5000 "
            "(relative, not robot XYZ).", messages
        )
        robot.move_linear_pose.assert_not_called()

    def test_missing_or_extra_markers_do_not_fall_back_to_fixed_roi(self):
        for corners in [
            [],
            [(100, 80), (540, 80), (540, 420)],
            [(100, 80), (540, 80), (540, 420), (100, 420), (600, 240)],
        ]:
            frame = workspace_frame(corners)
            cv2.rectangle(frame, (250, 150), (290, 190), (10, 10, 10), -1)
            detections, workspace = detector.detect_dice(frame)
            self.assertIsNone(workspace)
            self.assertEqual(detections, [])

    def test_perspective_workspace_rejects_outside_and_edge_objects(self):
        corners = [(130, 70), (500, 100), (550, 410), (80, 380)]
        frame = workspace_frame(corners)
        cv2.rectangle(frame, (300, 200), (330, 230), (10, 10, 10), -1)
        # Inside the bounding rectangle, but outside the slanted board edge.
        cv2.rectangle(frame, (85, 100), (110, 125), (10, 10, 10), -1)
        cv2.rectangle(frame, (95, 250), (125, 280), (10, 10, 10), -1)
        detections, workspace = detector.detect_dice(frame)
        self.assertIsNotNone(workspace)
        self.assertEqual(len(detections), 1)
        np.testing.assert_allclose(detections[0]["center"], (315, 215), atol=1)

    def test_workspace_follows_shifted_and_rotated_markers(self):
        for corners in [
            [(40, 40), (390, 40), (390, 330), (40, 330)],
            [(300, 30), (570, 230), (350, 440), (80, 220)],
        ]:
            frame = workspace_frame(corners)
            cv2.rectangle(frame, (240, 180), (270, 210), (10, 10, 10), -1)
            detections, workspace = detector.detect_dice(frame)
            self.assertIsNotNone(workspace)
            self.assertEqual(len(detections), 1)

    def test_blurred_markers(self):
        frame = cv2.GaussianBlur(workspace_frame(), (5, 5), 1.0)
        self.assertIsNotNone(detector.detect_workspace(frame))

    def test_blurred_low_contrast_markers_with_dark_surroundings(self):
        for sigma in (1, 2):
            with self.subTest(sigma=sigma):
                frame = workspace_frame()
                frame = (140 + frame.astype(np.float32) / 255 * 50).astype(np.uint8)
                frame[:, :50] = 15
                frame = cv2.GaussianBlur(frame, (11, 11), sigma)
                workspace = detector.detect_workspace(frame)
                self.assertIsNotNone(workspace)
                np.testing.assert_allclose(
                    workspace.corners,
                    [(100, 80), (540, 80), (540, 420), (100, 420)],
                    atol=1,
                )

    def test_pips_and_square_patterns_are_not_ring_markers(self):
        frame = workspace_frame([])
        for x, y in [(150, 100), (350, 100), (150, 300), (350, 300)]:
            for half_side, color in [(20, 10), (15, 240), (10, 10), (5, 240)]:
                cv2.rectangle(
                    frame, (x - half_side, y - half_side),
                    (x + half_side, y + half_side), (color, color, color), -1
                )
        self.assertIsNone(detector.detect_workspace(frame))

    def test_real_board_from_existing_project_screenshot(self):
        image = cv2.imread(str(MODULE_PATH.parents[1] / "extra" / "ahoj.png"))
        self.assertIsNotNone(image)
        # The existing screenshot contains the camera view in the upper-left window.
        frame = image[104:700, 56:854].copy()
        dice, workspace = detector.detect_dice(frame)
        self.assertIsNotNone(workspace)
        self.assertEqual(len(dice), 3)
        np.testing.assert_allclose(
            workspace.corners, [(228, 33), (551, 28), (568, 356), (222, 358)], atol=2
        )

    def test_degenerate_marker_geometry(self):
        for corners in [
            [(100, 100), (220, 100), (340, 100), (460, 100)],
            [(100, 80), (540, 80), (540, 420), (300, 200)],
        ]:
            self.assertIsNone(detector.detect_workspace(workspace_frame(corners)))

    def test_marker_loss_clears_previous_workspace(self):
        self.assertIsNotNone(detector.detect_workspace(workspace_frame()))
        detections, workspace = detector.detect_dice(workspace_frame([]))
        self.assertEqual(detections, [])
        self.assertIsNone(workspace)

    def test_missing_markers_block_motion_even_with_calibration(self):
        robot = Mock()
        frame = workspace_frame([])
        with (
            patch.object(sys, "argv", ["rectified_detector", "--enable-motion"]),
            patch.object(detector, "NiryoRobot", return_value=robot),
            patch.object(detector, "CALIBRATION_FILE") as file,
            patch.object(detector, "load_calibration", return_value=self.mapping),
            patch.object(detector, "read_frame", return_value=frame),
            patch.object(detector.cv2, "namedWindow"),
            patch.object(detector.cv2, "setMouseCallback"),
            patch.object(detector.cv2, "imshow"),
            patch.object(detector.cv2, "destroyAllWindows"),
            patch.object(detector.cv2, "waitKey", side_effect=[ord("m"), ord("q")]),
            patch.object(detector, "confirmed_move") as move,
        ):
            file.exists.return_value = True
            detector.main()
        move.assert_not_called()
        robot.move_linear_pose.assert_not_called()


if __name__ == "__main__":
    unittest.main()
