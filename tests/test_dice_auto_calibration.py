import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np


VISION = Path(__file__).resolve().parents[1] / "experiments" / "vision"
SPEC = importlib.util.spec_from_file_location(
    "hand_eye_auto_tests", VISION / "dice_hand_eye_calibration.py"
)
hand_eye = importlib.util.module_from_spec(SPEC)
with patch.object(sys, "path", [str(VISION), *sys.path]):
    SPEC.loader.exec_module(hand_eye)

RobotKinematics = hand_eye.RobotKinematics
read_robot_parameters = hand_eye.read_robot_parameters


def robot_description():
    links = ["base_link", "one", "two", "three", "four", "wrist_link", "hand_link"]
    axes = ["1 0 0", "0 1 0", "0 0 1", "1 0 0", "0 1 0", "0 0 1"]
    joints = []
    for index in range(6):
        joints.append(
            f'<joint name="joint_{index + 1}" type="revolute">'
            f'<parent link="{links[index]}"/><child link="{links[index + 1]}"/>'
            f'<origin xyz="0.01 0 0.03"/><axis xyz="{axes[index]}"/></joint>'
        )
    return '<robot name="test">' + "".join(joints) + (
        '<joint name="tool_fixed" type="fixed"><parent link="hand_link"/>'
        '<child link="tool_link"/><origin xyz="0 0 0.03"/></joint></robot>'
    )


def samples_for_mount(frame):
    model = RobotKinematics(robot_description())
    camera = hand_eye.pose_transform([0.03, 0.02, 0.05, 0.1, -0.2, 0.3])
    board = hand_eye.pose_transform([0.25, -0.1, 0, 0, 0, 0.1])
    angles = [
        [0, 0, 0, 0, 0, 0], [0.25, 0, 0, 0, 0, 0.3],
        [-0.25, 0, 0, 0, 0, -0.25], [0, 0.3, 0, 0, 0, 0.1],
        [0, -0.3, 0, 0, 0, -0.35], [0.15, 0.2, 0.1, 0, 0, 0.2],
        [0.12, -0.2, 0.1, 0, 0, -0.2], [-0.15, 0.12, -0.1, 0, 0, 0.25],
    ]
    samples = []
    for joints in angles:
        tool = model.forward(joints, "tool_link")
        wrist = model.forward(joints, "wrist_link")
        reference = wrist if frame == "wrist_link" else tool
        samples.append({
            "base_tool": tool.tolist(), "base_wrist": wrist.tolist(),
            "camera_board": (np.linalg.inv(camera) @ np.linalg.inv(reference) @ board).tolist(),
            "joints": joints, "image_size": [640, 480], "reprojection_rmse_px": 0.1,
        })
    return samples, model, camera, board


class AutomaticCalibrationTests(unittest.TestCase):
    def test_selects_both_physical_mounts_with_joint6_changes(self):
        for frame in ("tool_link", "wrist_link"):
            with self.subTest(frame=frame):
                samples, model, camera, board = samples_for_mount(frame)
                result = hand_eye.fit_automatic(samples, 0.17, model)
                self.assertEqual(result["reference_frame"], frame)
                np.testing.assert_allclose(result["reference_from_camera"], camera, atol=1e-8)
                np.testing.assert_allclose(result["base_from_board"], board, atol=1e-8)
                self.assertNotIn("required_joint6_rad", result)
                self.assertNotIn("tool_from_camera", result)
                self.assertFalse(result["motion_enabled"])

    def test_fixed_joint6_is_not_enough_to_identify_mount(self):
        samples, model, _, _ = samples_for_mount("wrist_link")
        # A failed model check or insufficient joint variation must block selection.
        for sample in samples:
            sample["joints"][5] = 0
        with self.assertRaises(ValueError):
            hand_eye.fit_automatic(samples, 0.17, model)

    def test_both_passing_mount_hypotheses_are_rejected(self):
        samples, model, _, _ = samples_for_mount("wrist_link")
        with patch.object(hand_eye, "fit_samples", return_value={"motion_enabled": False}):
            with self.assertRaisesRegex(ValueError, "Both camera mounts fit"):
                hand_eye.fit_automatic(samples, 0.17, model)

    def test_moved_board_and_invalid_saved_wrist_are_rejected(self):
        samples, model, _, _ = samples_for_mount("wrist_link")
        samples[-1]["camera_board"][0][3] += 0.03
        with self.assertRaisesRegex(ValueError, "Neither camera mount fits"):
            hand_eye.fit_automatic(samples, 0.17, model)
        samples, model, _, _ = samples_for_mount("tool_link")
        samples[0]["base_wrist"][0][3] += 0.001
        with self.assertRaisesRegex(ValueError, "Saved wrist pose"):
            hand_eye.fit_automatic(samples, 0.17, model)

    def test_model_disagreement_blocks_processing(self):
        model = RobotKinematics(robot_description())
        tool = model.forward([0] * 6, "tool_link")
        model.checked_wrist([0] * 6, tool)
        tool[2, 3] += 0.01
        with self.assertRaisesRegex(ValueError, "does not match"):
            model.checked_wrist([0] * 6, tool)

    def test_auto_capture_accepts_changed_joint6_and_records_wrist(self):
        samples, model, _, _ = samples_for_mount("wrist_link")
        current = samples[1]
        robot = Mock(spec=["get_pose", "get_joints"])
        robot.get_pose.return_value.to_list.return_value = [0] * 6
        robot.get_joints.return_value = current["joints"]
        workspace = SimpleNamespace(corners=np.array([[100, 80], [540, 80],
                                                      [540, 420], [100, 420]]))
        with (
            patch.object(hand_eye, "pose_transform", return_value=np.array(current["base_tool"])),
            patch.object(hand_eye, "read_frame", return_value=np.zeros((480, 640, 3), np.uint8)),
            patch.object(hand_eye, "detect_workspace", return_value=workspace),
            patch.object(hand_eye, "estimate_board_pose",
                         return_value=(np.array(current["camera_board"]), 0.1)),
            patch.object(hand_eye.time, "sleep"),
        ):
            result = hand_eye.capture_sample(
                robot, (np.eye(3), np.zeros(5)), 0.17, [samples[0]], model
            )
        self.assertEqual(result["joints"][5], 0.3)
        np.testing.assert_allclose(result["base_wrist"], current["base_wrist"])

    def test_joint6_changes_tool_but_not_wrist(self):
        model = RobotKinematics(robot_description())
        first = [0] * 6
        second = [0, 0, 0, 0, 0, 0.5]
        np.testing.assert_allclose(
            model.forward(first, "wrist_link"), model.forward(second, "wrist_link")
        )
        self.assertFalse(np.allclose(
            model.forward(first, "tool_link"), model.forward(second, "tool_link")
        ))

    def test_urdf_origin_rotation_and_axis_are_applied_in_order(self):
        description = robot_description().replace(
            '<origin xyz="0.01 0 0.03"/><axis xyz="1 0 0"/>',
            '<origin xyz="0.01 0 0.03" rpy="0 0 1.5707963267948966"/>'
            '<axis xyz="1 0 0"/>', 1
        )
        model = RobotKinematics(description)
        first_link = model.forward([0.2, 0, 0, 0, 0, 0], "one")
        expected = hand_eye.pose_transform([0.01, 0, 0.03, 0.2, 0, np.pi / 2])
        np.testing.assert_allclose(first_link, expected, atol=1e-8)

    def test_stability_timer_and_once_per_stop(self):
        timer = hand_eye.StableView()
        pose = np.eye(4)
        joints = np.zeros(6)
        corners = np.array([[0, 0], [100, 0], [100, 100], [0, 100]])
        self.assertFalse(timer.ready(pose, joints, corners, 0))
        self.assertFalse(timer.ready(pose, joints, corners, 1))
        self.assertTrue(timer.ready(pose, joints, corners, 1.6))
        self.assertFalse(timer.ready(pose, joints, corners, 3))
        changed = pose.copy()
        changed[0, 3] += 0.01
        self.assertFalse(timer.ready(changed, joints, corners, 4))
        self.assertTrue(timer.ready(changed, joints, corners, 5.6))
        self.assertFalse(timer.ready(changed, joints, None, 6))
        self.assertFalse(timer.ready(changed, joints, corners, 7))
        self.assertTrue(timer.ready(changed, joints, corners, 8.6))

    def test_motion_and_marker_changes_reset_stability(self):
        timer = hand_eye.StableView()
        pose = np.eye(4)
        joints = np.zeros(6)
        corners = np.zeros((4, 2))
        timer.ready(pose, joints, corners, 0)
        changed = joints.copy()
        changed[5] = 0.1
        self.assertFalse(timer.ready(pose, changed, corners, 1.6))
        self.assertFalse(timer.ready(pose, changed, corners + 3, 3))
        self.assertTrue(timer.ready(pose, changed, corners + 3, 4.6))

    def test_parameter_query_uses_read_only_ros_calls_and_closes(self):
        connection = Mock()
        connection.recv.side_effect = [
            json.dumps({"op": "service_response", "id": "hand-eye-read-0", "result": True,
                        "values": {"value": json.dumps(robot_description())}}),
            json.dumps({"op": "service_response", "id": "hand-eye-read-1", "result": True,
                        "values": {"value": "false"}}),
        ]
        with patch("websocket.create_connection", return_value=connection):
            values = read_robot_parameters("127.0.0.1")
        self.assertFalse(values["/niryo_robot_vision/undistort_stream"])
        for call in connection.send.call_args_list:
            request = json.loads(call.args[0])
            self.assertEqual(request["service"], "/rosapi/get_param")
        connection.close.assert_called_once()

    def test_ros_error_is_not_silently_ignored(self):
        connection = Mock()
        connection.recv.return_value = json.dumps({"op": "status", "level": "error",
                                                    "msg": "service unavailable"})
        with patch("websocket.create_connection", return_value=connection):
            with self.assertRaisesRegex(ValueError, "service unavailable"):
                read_robot_parameters("127.0.0.1")
        connection.close.assert_called_once()

    def test_auto_session_records_and_fits_without_motion_methods(self):
        samples, model, _, _ = samples_for_mount("wrist_link")
        robot = Mock(spec=["get_pose", "get_joints", "get_hardware_status",
                           "get_camera_intrinsics", "close_connection"])
        robot.get_hardware_status.return_value = SimpleNamespace(
            calibration_needed=False, calibration_in_progress=False, hardware_version="ned"
        )
        robot.get_camera_intrinsics.return_value = (np.eye(3), np.zeros(5))
        robot.get_joints.return_value = [0] * 6
        robot.get_pose.return_value.to_list.return_value = [0] * 6
        parameters = {"/robot_description": model.description,
                      "/niryo_robot_vision/undistort_stream": False}
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "result.json"
            with (
                patch.object(sys, "argv", ["hand_eye", "--auto"]),
                patch("builtins.input", return_value="CAPTURE"),
                patch.object(hand_eye, "NiryoRobot", return_value=robot),
                patch.object(hand_eye, "read_robot_parameters", return_value=parameters),
                patch.object(hand_eye, "pose_transform",
                             return_value=model.forward([0] * 6, "tool_link")),
                patch.object(hand_eye, "read_frame",
                             return_value=np.zeros((480, 640, 3), np.uint8)),
                patch.object(hand_eye, "detect_workspace", return_value=None),
                patch.object(hand_eye.StableView, "ready", return_value=True),
                patch.object(hand_eye, "capture_sample", side_effect=copy.deepcopy(samples)),
                patch.object(hand_eye, "OUTPUT", output),
                patch.object(hand_eye, "SAMPLES", Path(directory) / "draft.json"),
                patch.object(hand_eye.cv2, "imshow"),
                patch.object(hand_eye.cv2, "destroyAllWindows"),
                patch.object(hand_eye.cv2, "waitKey", return_value=255),
            ):
                hand_eye.main()
            result = json.loads(output.read_text())
            self.assertEqual(result["reference_frame"], "wrist_link")
            self.assertEqual(result["capture_mode"], "automatic_read_only")
            self.assertFalse(result["motion_enabled"])
            self.assertEqual(len(result["samples"]), 8)
        robot.close_connection.assert_called_once()


if __name__ == "__main__":
    unittest.main()
