import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from test_dice_camera_calibration import sample_data

spec = importlib.util.spec_from_file_location('empty_approach', Path(__file__).resolve().parents[1] / 'diagnostics' / 'check_empty_approach.py')
check = importlib.util.module_from_spec(spec)
spec.loader.exec_module(check)


class EmptyApproachTests(unittest.TestCase):
    def setUp(self):
        self.data = sample_data()
        self.pose = list(self.data['overview'])
        self.robot = Mock()
        self.robot.get_pose.return_value.to_list.side_effect = lambda: list(self.pose)
        self.robot.get_learning_mode.return_value = False
        self.robot.get_hardware_status.return_value = SimpleNamespace(
            connection_up=True, calibration_needed=False, calibration_in_progress=False,
            motor_names=['Tool'], hardware_errors=[0], error_message='')

    def test_cancel_never_actuates(self):
        with patch.object(check, 'move_linear_pose_compatible') as move:
            result = check.test_empty_approach(self.robot, self.data, lambda _: 'cancel')
        self.assertIsNone(result)
        move.assert_not_called()
        self.robot.set_arm_max_velocity.assert_not_called()

    def test_success_stays_high_and_never_actuates_gripper(self):
        def move(_robot, target):
            self.pose = target.tolist()
        with (tempfile.TemporaryDirectory() as directory,
              patch.object(check, 'REPORT_FILE', Path(directory) / 'report.json'),
              patch.object(check, 'move_linear_pose_compatible', side_effect=move) as motion,
              patch.object(check, 'wait_for_pose') as wait):
            report = check.test_empty_approach(self.robot, self.data, lambda _: 'TEST EMPTY ARM')
            saved = json.loads(check.REPORT_FILE.read_text())
        self.assertEqual(motion.call_count, 2)
        for call in wait.call_args_list:
            self.assertEqual(call.kwargs, {'angle_tolerance': check.OVERVIEW_ANGLE_TOLERANCE})
        self.assertEqual(saved, report)
        self.assertEqual(report['target'][2], report['start'][2])
        self.assertEqual(self.pose, self.data['overview'])
        self.robot.open_gripper.assert_not_called()
        self.robot.close_gripper.assert_not_called()
        self.robot.tool_reboot.assert_not_called()

    def test_failed_approach_does_not_return_automatically(self):
        with (tempfile.TemporaryDirectory() as directory,
              patch.object(check, 'REPORT_FILE', Path(directory) / 'report.json'),
              patch.object(check, 'move_linear_pose_compatible') as motion,
              patch.object(check, 'wait_for_pose', side_effect=ValueError('Endpoint mismatch'))):
            with self.assertRaisesRegex(ValueError, 'Endpoint mismatch'):
                check.test_empty_approach(self.robot, self.data, lambda _: 'TEST EMPTY ARM')
        motion.assert_called_once()
        self.robot.open_gripper.assert_not_called()
        self.robot.close_gripper.assert_not_called()

    def test_low_start_blocks_before_confirmation(self):
        self.pose[2] = .2
        confirm = Mock()
        with patch.object(check, 'move_linear_pose_compatible') as motion:
            with self.assertRaises(ValueError):
                check.test_empty_approach(self.robot, self.data, confirm)
        confirm.assert_not_called()
        motion.assert_not_called()
