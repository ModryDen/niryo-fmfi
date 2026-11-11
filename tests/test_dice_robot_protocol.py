import unittest
from unittest.mock import patch, Mock
import json
from pyniryo import NiryoRobot
from pyniryo.api.enums_communication import Command
from pyniryo.api.exceptions import NiryoRobotException
from test_dice_camera_calibration import calibration


class LinearPoseProtocolTests(unittest.TestCase):
    def setUp(self):
        self.robot = NiryoRobot()  # No connection, every transport call is intercepted.
        self.pose = [.21, .01, .4, -.15, 1.5, -.01]

    def test_installed_sdk_rejection_switches_to_exactly_six_parameters(self):
        packets = []
        def server(command, *args):
            packets.append((command, args))
            if len(args) == 7:
                raise NiryoRobotException('Command KO : 6 parameters expected but 7 given\nIn function')
            self.assertEqual(len(args), 6)
        with patch.object(self.robot, '_NiryoRobot__send_n_receive', side_effect=server):
            calibration.move_linear_pose_compatible(self.robot, self.pose)
            calibration.move_linear_pose_compatible(self.robot, self.pose)
        self.assertEqual([len(args) for _, args in packets], [7, 6, 6])
        self.assertTrue(all(command == Command.MOVE_LINEAR_POSE for command, _ in packets))
        self.assertEqual(list(packets[1][1]), self.pose)
        self.assertEqual(list(packets[2][1]), self.pose)
        self.assertEqual(packets[0][1][-1], '')

    def test_modern_server_keeps_standard_sdk_protocol(self):
        with patch.object(self.robot, '_NiryoRobot__send_n_receive') as send:
            calibration.move_linear_pose_compatible(self.robot, self.pose)
        send.assert_called_once_with(Command.MOVE_LINEAR_POSE, *self.pose, '')

    def test_motion_errors_are_never_retried(self):
        for error in (NiryoRobotException('Command KO : inverse kinematics failed'),
                      TimeoutError('network timeout')):
            with self.subTest(error=error):
                with patch.object(self.robot, '_NiryoRobot__send_n_receive', side_effect=error) as send:
                    with self.assertRaises(type(error)):
                        calibration.move_linear_pose_compatible(self.robot, self.pose)
                self.assertEqual(send.call_count, 1)

    def test_failure_of_six_parameter_command_propagates_without_another_retry(self):
        with patch.object(self.robot, '_NiryoRobot__send_n_receive', side_effect=[
            NiryoRobotException('6 parameters expected but 7 given'),
            NiryoRobotException('inverse kinematics failed'),
        ]) as send:
            with self.assertRaisesRegex(NiryoRobotException, 'inverse kinematics'):
                calibration.move_linear_pose_compatible(self.robot, self.pose)
        self.assertEqual(send.call_count, 2)
        self.assertFalse(getattr(self.robot, '_dice_legacy_linear_pose', False))

    def test_wait_for_pose_accepts_delayed_feedback_without_resending(self):
        wrong = list(self.pose)
        wrong[0] -= .005
        with (patch.object(self.robot, 'get_pose') as get,
              patch.object(self.robot, '_NiryoRobot__send_n_receive') as send):
            get.return_value.to_list.side_effect = [wrong, self.pose, self.pose]
            calibration.wait_for_pose(self.robot, self.pose, interval=0)
        self.assertEqual(get.return_value.to_list.call_count, 3)
        send.assert_not_called()

    def test_wait_for_pose_rejects_persistent_error_without_resending(self):
        wrong = list(self.pose)
        wrong[0] -= .005
        with (patch.object(self.robot, 'get_pose') as get,
              patch.object(self.robot, '_NiryoRobot__send_n_receive') as send):
            get.return_value.to_list.return_value = wrong
            with self.assertRaisesRegex(ValueError, 'did not settle'):
                calibration.wait_for_pose(self.robot, self.pose, timeout=0)
        send.assert_not_called()

    def test_camera_return_angle_tolerance_does_not_relax_grasp_or_position(self):
        import numpy as np
        actual = list(self.pose)
        actual[5] += np.deg2rad(1.46)
        with self.assertRaises(ValueError):
            calibration.require_pose(actual, self.pose)
        calibration.require_pose(actual, self.pose, angle_tolerance=calibration.OVERVIEW_ANGLE_TOLERANCE)
        actual[0] += .015
        with self.assertRaises(ValueError):
            calibration.require_pose(actual, self.pose, angle_tolerance=calibration.OVERVIEW_ANGLE_TOLERANCE)

    def test_inactive_camera_is_rejected_even_if_tcp_could_return_cached_jpeg(self):
        connection = Mock()
        connection.recv.return_value = json.dumps(dict(
            topic='/niryo_robot_vision/video_stream_is_active', msg={'data': False}))
        with patch('websocket.create_connection', return_value=connection):
            with self.assertRaisesRegex(ValueError, 'stream is OFF'):
                calibration.require_live_camera('robot')
        connection.close.assert_called_once()
        for call in connection.send.call_args_list:
            self.assertEqual(json.loads(call.args[0])['op'], 'subscribe')

    def test_live_camera_requires_active_stream_and_different_frame_headers(self):
        active = dict(topic='/niryo_robot_vision/video_stream_is_active', msg={'data': True})
        def frame(seq):
            return dict(topic='/niryo_robot_vision/compressed_video_stream',
                        msg={'header': {'seq': seq, 'stamp': {'secs': 100, 'nsecs': seq}}})
        connection = Mock()
        connection.recv.side_effect = [json.dumps(x) for x in (active, frame(1), frame(1), frame(2))]
        with patch('websocket.create_connection', return_value=connection):
            calibration.require_live_camera('robot')
        self.assertEqual(connection.recv.call_count, 4)
        connection.close.assert_called_once()

    def test_invalid_pose_sends_nothing(self):
        with patch.object(self.robot, '_NiryoRobot__send_n_receive') as send:
            with self.assertRaises(ValueError):
                calibration.move_linear_pose_compatible(self.robot, [0] * 7)
        send.assert_not_called()


if __name__ == '__main__':
    unittest.main()
