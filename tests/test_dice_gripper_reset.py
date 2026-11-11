import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock
from pyniryo import ToolID

path = Path(__file__).resolve().parents[1] / 'diagnostics' / 'reset_gripper.py'
spec = importlib.util.spec_from_file_location('reset_gripper', path)
reset = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reset)


class GripperResetTests(unittest.TestCase):
    def robot(self):
        robot = Mock()
        robot.get_current_tool_id.return_value = ToolID.GRIPPER_1
        robot.get_hardware_status.return_value = SimpleNamespace(
            motor_names=['joint_1', 'Tool'], hardware_errors=[0, 0],
            connection_up=True, calibration_in_progress=False, error_message='')
        return robot

    def test_cancel_does_not_reset_or_move(self):
        robot = self.robot()
        self.assertFalse(reset.reset_gripper(robot, confirm=lambda _: 'cancel'))
        robot.tool_reboot.assert_not_called()
        robot.move_linear_pose.assert_not_called()
        robot.open_gripper.assert_not_called()
        robot.close_gripper.assert_not_called()

    def test_confirm_resets_only_tool(self):
        robot = self.robot()
        self.assertTrue(reset.reset_gripper(robot, confirm=lambda _: 'RESET EMPTY GRIPPER'))
        robot.tool_reboot.assert_called_once()
        robot.move_linear_pose.assert_not_called()
        robot.move_pose.assert_not_called()
        robot.open_gripper.assert_not_called()
        robot.close_gripper.assert_not_called()

    def test_arm_error_blocks_reset(self):
        robot = self.robot()
        robot.get_hardware_status.return_value.hardware_errors[0] = 1
        with self.assertRaises(ValueError):
            reset.reset_gripper(robot, confirm=lambda _: 'RESET EMPTY GRIPPER')
        robot.tool_reboot.assert_not_called()
