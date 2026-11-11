"""Confirmed empty-gripper approach test. No descent or gripper commands."""
from pathlib import Path
import json
import sys

import numpy as np
from pyniryo import NiryoRobot

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'experiments' / 'vision'))
from dice_camera_calibration import (
    pose_array, require_pose, wait_for_pose, orientation_distance,
    move_linear_pose_compatible, ANGLE_TOLERANCE, OVERVIEW_ANGLE_TOLERANCE,
)
from dice_pick_place import PICK_DRAFT_FILE, require_gripper_healthy

ROBOT_IP = '169.254.200.200'
REPORT_FILE = Path(__file__).with_name('empty_approach_report.json')


def test_empty_approach(robot, data, confirm=input):
    start = pose_array(robot.get_pose().to_list())
    if robot.get_learning_mode():
        raise ValueError('Disable Learning Mode in Studio first.')
    require_gripper_healthy(robot)
    if start[2] < .30:
        raise ValueError('Use Studio to reach a clear overview above Z=0.30 m first.')
    reference = pose_array(data['overview'])
    if np.linalg.norm(start[:3] - reference[:3]) > .02:
        raise ValueError('Start near the saved camera overview (within 20 mm).')
    target = pose_array(data['poses'][4])
    target[2] = start[2]  # Stay at the current overview height throughout.
    if np.linalg.norm(target[:2] - start[:2]) > .20:
        raise ValueError('Unexpected approach distance; inspect the saved target.')
    print('EMPTY GRIPPER TEST: approach above target 5 and return. No descent, opening or closing.')
    print('Start XYZ/RPY:', np.round(start, 6).tolist())
    print('Upper target XYZ/RPY:', np.round(target, 6).tolist())
    print('Check that the gripper is empty and the WHOLE arm path is clear. Speed stays at 10%.')
    if confirm('Type TEST EMPTY ARM to move, anything else cancels: ').strip() != 'TEST EMPTY ARM':
        return None
    require_pose(robot.get_pose().to_list(), start)
    require_gripper_healthy(robot)
    if robot.get_learning_mode():
        raise ValueError('Learning mode changed; test cancelled.')
    robot.set_arm_max_velocity(10)
    report = dict(start=start.tolist(), target=target.tolist(), stages=[])
    for name, goal in [('upper approach', target), ('return to start', start)]:
        print(f'Executing: {name}', flush=True)
        tolerance = OVERVIEW_ANGLE_TOLERANCE  # Both legs are upper movements; no grip or descent.
        stage = dict(name=name, requested=goal.tolist(), angle_limit_deg=float(np.degrees(tolerance)))
        report['stages'].append(stage)
        try:
            move_linear_pose_compatible(robot, goal)
            wait_for_pose(robot, goal, angle_tolerance=tolerance)
            require_gripper_healthy(robot)
            actual = pose_array(robot.get_pose().to_list())
            require_pose(actual, goal, angle_tolerance=tolerance)
            stage.update(actual=actual.tolist(), position_error_mm=float(np.linalg.norm(actual[:3]-goal[:3])*1000),
                         angle_error_deg=float(np.degrees(orientation_distance(actual, goal))), status='passed')
            print(f"PASS: {stage['position_error_mm']:.2f} mm, {stage['angle_error_deg']:.2f} degrees.", flush=True)
        except Exception as error:
            stage.update(status='failed', error=str(error))
            REPORT_FILE.write_text(json.dumps(report, indent=2), encoding='utf-8')
            print('Test stopped. No automatic recovery movement. Report:', REPORT_FILE)
            raise
    REPORT_FILE.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('EMPTY APPROACH PASSED. Robot returned to the starting pose.')
    return report


def main():
    robot = None
    try:
        data = json.loads(PICK_DRAFT_FILE.read_text(encoding='utf-8'))
        robot = NiryoRobot(ROBOT_IP)
        test_empty_approach(robot, data)
    except Exception as error:
        print(f'Test stopped: {type(error).__name__}: {error}')
    finally:
        if robot is not None:
            robot.close_connection()


if __name__ == '__main__':
    main()
