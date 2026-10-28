import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np


VISION = Path(__file__).resolve().parents[1] / "experiments" / "vision"
SPEC = importlib.util.spec_from_file_location(
    "dice_hand_eye_calibration", VISION / "dice_hand_eye_calibration.py"
)
hand_eye = importlib.util.module_from_spec(SPEC)
with patch.object(sys, "path", [str(VISION), *sys.path]):
    SPEC.loader.exec_module(hand_eye)


def rotation(vector):
    return cv2.Rodrigues(np.asarray(vector, dtype=float))[0]


def observations():
    tool_camera = hand_eye.transform(rotation([0.1, -0.2, 0.3]), [0.03, 0.02, 0.05])
    base_board = hand_eye.transform(rotation([0.0, 0.0, 0.1]), [0.25, -0.1, 0])
    angles = [
        [0, 0, 0], [0.2, 0, 0], [-0.2, 0, 0], [0, 0.25, 0],
        [0, -0.25, 0], [0.1, 0.2, 0.1], [0.12, -0.15, 0.1], [-0.1, 0.12, -0.1],
    ]
    samples = []
    for index, angle in enumerate(angles):
        base_tool = hand_eye.transform(
            rotation(angle), [0.12 + index * 0.01, np.sin(index) * 0.02, 0.4]
        )
        camera_board = np.linalg.inv(tool_camera) @ np.linalg.inv(base_tool) @ base_board
        samples.append({
            "base_tool": base_tool.tolist(), "camera_board": camera_board.tolist(),
            "joints": [0, 0, 0, 0, 0, 0.1],
            "image_size": [640, 480], "reprojection_rmse_px": 0.1,
        })
    return samples, tool_camera, base_board


class HandEyeTests(unittest.TestCase):
    def test_recovers_transform_direction_and_metric_scale(self):
        samples, expected_camera, expected_board = observations()
        result = hand_eye.fit_samples(samples, 0.17)
        np.testing.assert_allclose(result["tool_from_camera"], expected_camera, atol=1e-8)
        np.testing.assert_allclose(result["base_from_board"], expected_board, atol=1e-8)
        self.assertEqual(result["training_views"], 6)
        self.assertEqual(result["validation_views"], 2)
        self.assertFalse(result["motion_enabled"])
        self.assertEqual(result["status"], "geometry_checked_not_motion_approved")

    def test_small_observation_noise_passes_geometric_checks(self):
        samples, expected_camera, _ = observations()
        rng = np.random.default_rng(42)
        for sample in samples:
            matrix = np.array(sample["camera_board"])
            matrix[:3, :3] = rotation(rng.normal(0, 0.0002, 3)) @ matrix[:3, :3]
            matrix[:3, 3] += rng.normal(0, 0.0001, 3)
            sample["camera_board"] = matrix.tolist()
        result = hand_eye.fit_samples(samples, 0.17)
        np.testing.assert_allclose(result["tool_from_camera"], expected_camera, atol=0.003)

    def test_requires_eight_views(self):
        samples, _, _ = observations()
        with self.assertRaisesRegex(ValueError, "at least 8"):
            hand_eye.fit_samples(samples[:7], 0.17)

    def test_no_rotation_or_single_axis_is_rejected(self):
        samples, camera, board = observations()
        for axis in (None, [0, 0, 1]):
            data = copy.deepcopy(samples)
            for index, sample in enumerate(data):
                angle = np.zeros(3) if axis is None else np.array(axis) * index * 0.08
                tool = hand_eye.transform(rotation(angle), [0.12 + 0.01 * index, 0, 0.4])
                sample["base_tool"] = tool.tolist()
                sample["camera_board"] = (np.linalg.inv(camera) @ np.linalg.inv(tool) @ board).tolist()
            with self.assertRaises(ValueError):
                hand_eye.fit_samples(data, 0.17)

    def test_validation_views_do_not_affect_fit_and_detect_moved_board(self):
        samples, _, _ = observations()
        samples[-1]["camera_board"][0][3] += 0.02
        with self.assertRaisesRegex(ValueError, "View 8 inconsistent"):
            hand_eye.fit_samples(samples, 0.17)

    def test_reject_joint6_change_and_nonrigid_matrices(self):
        samples, _, _ = observations()
        samples[-1]["joints"][5] += 0.1
        with self.assertRaisesRegex(ValueError, "Joint 6 changed"):
            hand_eye.fit_samples(samples, 0.17)
        samples, _, _ = observations()
        samples[0]["base_tool"][0][0] = 2
        with self.assertRaisesRegex(ValueError, "rigid transform"):
            hand_eye.fit_samples(samples, 0.17)

    def test_board_pose_from_known_projection(self):
        intrinsics = np.array([[508, 0, 333], [0, 506, 243], [0, 0, 1]], dtype=float)
        rvec = np.array([0.2, -0.15, 0.03])
        translation = np.array([-0.085, -0.085, 0.4])
        corners = cv2.projectPoints(
            hand_eye.board_points(0.17), rvec, translation, intrinsics, np.zeros(5)
        )[0].reshape(-1, 2)
        pose, error = hand_eye.estimate_board_pose(corners, intrinsics, 0.17)
        np.testing.assert_allclose(pose[:3, 3], translation, atol=1e-8)
        np.testing.assert_allclose(pose[:3, :3], rotation(rvec), atol=1e-8)
        self.assertLess(error, 1e-8)

    def test_robot_rpy_rotation_convention(self):
        pose = hand_eye.pose_transform([0.1, 0.2, 0.3, 0, 0, np.pi / 2])
        np.testing.assert_allclose(pose[:3, 3], [0.1, 0.2, 0.3])
        np.testing.assert_allclose(pose[:3, :3] @ [1, 0, 0], [0, 1, 0], atol=1e-8)

    def test_holdout_corner_orientation_change_is_rejected(self):
        samples, _, _ = observations()
        pose = np.asarray(samples[-1]["camera_board"])
        pose[:3, :3] = pose[:3, :3] @ rotation([0, 0, np.pi / 2])
        samples[-1]["camera_board"] = pose.tolist()
        with self.assertRaisesRegex(ValueError, "View 8 inconsistent"):
            hand_eye.fit_samples(samples, 0.17)

    def test_invalid_board_size_and_input(self):
        for size in (0, -0.17, float("nan"), 170):
            with self.assertRaises(ValueError):
                hand_eye.board_points(size)
        with self.assertRaisesRegex(ValueError, "marker centres"):
            hand_eye.estimate_board_pose(np.zeros((3, 2)), np.eye(3), 0.17)

    def test_moving_robot_capture_is_rejected(self):
        robot = Mock(spec=["get_pose", "get_joints"])
        robot.get_pose.side_effect = [
            SimpleNamespace(to_list=lambda: [0, 0, 0.4, 0, 0, 0]),
            SimpleNamespace(to_list=lambda: [0, 0, 0.405, 0, 0, 0]),
        ]
        robot.get_joints.return_value = [0] * 6
        with (
            patch.object(hand_eye, "read_frame", return_value=np.zeros((480, 640, 3), np.uint8)),
            patch.object(hand_eye, "detect_workspace", return_value=None),
            patch.object(hand_eye.time, "sleep"),
        ):
            with self.assertRaisesRegex(ValueError, "Robot is moving"):
                hand_eye.capture_sample(robot, (np.eye(3), np.zeros(5)), 0.17, [])

    def test_stable_capture_records_reads_and_rejects_duplicates(self):
        robot = Mock(spec=["get_pose", "get_joints"])
        robot.get_pose.return_value = SimpleNamespace(
            to_list=lambda: [0.12, 0, 0.4, 0, 1.5, 0]
        )
        robot.get_joints.return_value = [0] * 6
        workspace = SimpleNamespace(corners=np.array([[100, 80], [540, 80],
                                                      [540, 420], [100, 420]]))
        with (
            patch.object(hand_eye, "read_frame", return_value=np.zeros((480, 640, 3), np.uint8)),
            patch.object(hand_eye, "detect_workspace", return_value=workspace),
            patch.object(hand_eye, "estimate_board_pose", return_value=(np.eye(4), 0.1)),
            patch.object(hand_eye.time, "sleep"),
        ):
            sample = hand_eye.capture_sample(robot, (np.eye(3), np.zeros(5)), 0.17, [])
            self.assertEqual(sample["image_size"], [640, 480])
            self.assertEqual(sample["reprojection_rmse_px"], 0.1)
            with self.assertRaisesRegex(ValueError, "Duplicate view"):
                hand_eye.capture_sample(robot, (np.eye(3), np.zeros(5)), 0.17, [sample])
            changed = copy.deepcopy(sample)
            changed["joints"][5] = 0.1
            with self.assertRaisesRegex(ValueError, "Joint 6 changed"):
                hand_eye.capture_sample(robot, (np.eye(3), np.zeros(5)), 0.17, [changed])

    def test_cancel_does_not_connect(self):
        with (
            patch.object(sys, "argv", ["hand_eye"]),
            patch("builtins.input", return_value="cancel"),
            patch.object(hand_eye, "NiryoRobot") as connect,
        ):
            hand_eye.main()
        connect.assert_not_called()

    def test_record_fit_and_save_use_only_read_methods(self):
        samples, _, _ = observations()
        robot = Mock(spec=["get_hardware_status", "get_camera_intrinsics",
                           "get_joints", "close_connection"])
        robot.get_hardware_status.return_value = SimpleNamespace(
            calibration_needed=False, calibration_in_progress=False, hardware_version="ned"
        )
        robot.get_camera_intrinsics.return_value = (np.eye(3), np.zeros(5))
        robot.get_joints.return_value = [0, 0, 0, 0, 0, 0.1]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "calibration.json"
            sample_path = Path(directory) / "samples.json"
            with (
                patch.object(sys, "argv", ["hand_eye"]),
                patch("builtins.input", return_value="CAPTURE"),
                patch.object(hand_eye, "NiryoRobot", return_value=robot),
                patch.object(hand_eye, "read_frame", return_value=np.zeros((480, 640, 3), np.uint8)),
                patch.object(hand_eye, "detect_workspace", return_value=None),
                patch.object(hand_eye, "capture_sample", side_effect=samples),
                patch.object(hand_eye, "OUTPUT", output),
                patch.object(hand_eye, "SAMPLES", sample_path),
                patch.object(hand_eye.cv2, "imshow"),
                patch.object(hand_eye.cv2, "destroyAllWindows"),
                patch.object(hand_eye.cv2, "waitKey", side_effect=[ord("c")] * 8 + [ord("f")]),
            ):
                hand_eye.main()
            saved = json.loads(output.read_text(encoding="utf-8"))
            self.assertFalse(saved["motion_enabled"])
            self.assertEqual(len(saved["samples"]), 8)
            draft = json.loads(sample_path.read_text(encoding="utf-8"))
            self.assertEqual(draft["status"], "unvalidated")
            self.assertEqual(len(draft["samples"]), 8)
            self.assertFalse(output.with_suffix(".json.tmp").exists())
        robot.close_connection.assert_called_once()

    def test_failed_fit_does_not_overwrite_previous_result(self):
        robot = Mock()
        robot.get_hardware_status.return_value = SimpleNamespace(
            calibration_needed=False, calibration_in_progress=False, hardware_version="ned"
        )
        robot.get_camera_intrinsics.return_value = (np.eye(3), np.zeros(5))
        robot.get_joints.return_value = [0] * 6
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "calibration.json"
            output.write_text('{"previous": true}', encoding="utf-8")
            with (
                patch.object(sys, "argv", ["hand_eye"]),
                patch("builtins.input", return_value="CAPTURE"),
                patch.object(hand_eye, "NiryoRobot", return_value=robot),
                patch.object(hand_eye, "read_frame", return_value=np.zeros((480, 640, 3), np.uint8)),
                patch.object(hand_eye, "detect_workspace", return_value=None),
                patch.object(hand_eye, "OUTPUT", output),
                patch.object(hand_eye, "SAMPLES", Path(directory) / "samples.json"),
                patch.object(hand_eye.cv2, "imshow"),
                patch.object(hand_eye.cv2, "destroyAllWindows"),
                patch.object(hand_eye.cv2, "waitKey", side_effect=[ord("f"), ord("q")]),
            ):
                hand_eye.main()
            self.assertEqual(json.loads(output.read_text()), {"previous": True})


if __name__ == "__main__":
    unittest.main()
