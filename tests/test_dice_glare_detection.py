import contextlib
import io
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np
from test_dice_camera_calibration import detector
from test_dice_area_pick import area_data
from test_dice_pick_place import pickup

FRAME = Path(__file__).with_name('fixtures') / 'dice_glare_frame.png'


class GlareDetectionTests(unittest.TestCase):
    def test_real_glare_frame_recovers_outline_without_losing_other_dice(self):
        frame = cv2.imread(str(FRAME))
        dice, workspace = detector.detect_dice(frame)
        self.assertIsNotNone(workspace)
        self.assertEqual(len(dice), 5)
        right = min(dice, key=lambda d: np.linalg.norm(np.asarray(d['center'])-[395,181]))
        self.assertTrue(right['glare_estimate'])
        # Approximate visible face centre, not the old dark-side centre (388,186).
        self.assertLess(np.linalg.norm(np.asarray(right['center'])-[395,181]), 4)
        self.assertGreater(right['center'][0], 391)
        self.assertLess(right['center'][1], 184)
        self.assertEqual(sum(bool(d['glare_estimate']) for d in dice), 1)

    def test_second_real_frame_recovers_missing_bright_die(self):
        frame = cv2.imread(str(FRAME.with_name('dice_glare_second.png')))
        dice, workspace = detector.detect_dice(frame)
        self.assertIsNotNone(workspace)
        self.assertEqual(len(dice),5)
        right = min(dice,key=lambda d: np.linalg.norm(np.asarray(d['center'])-[389,128]))
        self.assertTrue(right['glare_estimate'])
        self.assertLess(np.linalg.norm(np.asarray(right['center'])-[389,128]),3)

    def test_glare_estimate_cannot_start_full_pick(self):
        frame = cv2.imread(str(FRAME))
        robot = Mock()
        def click(_name, callback):
            callback(cv2.EVENT_LBUTTONDOWN, 394,182,0,None)
        with (
            patch.object(sys, 'argv', ['detector','--enable-motion']),
            patch.object(detector, 'NiryoRobot', return_value=robot),
            patch.object(detector, 'PICK_FILE') as file,
            patch.object(detector, 'load_pick_calibration', return_value=pickup.PickCalibration(area_data())),
            patch.object(detector, 'load_calibration', return_value=Mock()),
            patch.object(detector, 'read_frame', return_value=frame),
            patch.object(detector.cv2, 'namedWindow'),
            patch.object(detector.cv2, 'setMouseCallback', side_effect=click),
            patch.object(detector.cv2, 'imshow'),
            patch.object(detector.cv2, 'destroyAllWindows'),
            patch.object(detector.cv2, 'waitKey', side_effect=[ord('p'),ord('q')]),
            patch.object(detector, 'pick_and_place') as pick,
            patch.object(detector, 'require_live_camera') as camera,
            contextlib.redirect_stdout(io.StringIO()) as output,
        ):
            file.exists.return_value = True
            detector.main()
        self.assertIn('glare makes this die centre uncertain', output.getvalue())
        pick.assert_not_called()
        camera.assert_not_called()
        robot.move_linear_pose.assert_not_called()
        robot.close_gripper.assert_not_called()
