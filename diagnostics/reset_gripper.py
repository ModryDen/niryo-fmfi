"""Reset the gripper motor after explicit confirmation; never move the arm."""
import time

from pyniryo import NiryoRobot

ROBOT_IP = '169.254.200.200'


def reset_gripper(robot, confirm=input):
    tool = robot.get_current_tool_id()
    if not tool.name.startswith('GRIPPER_'):
        raise ValueError('Studio must identify the attached gripper first (SCAN).')
    status = robot.get_hardware_status()
    arm_errors = [(name, code) for name, code in zip(status.motor_names, status.hardware_errors)
                  if name != 'Tool' and code != 0]
    if arm_errors or not status.connection_up or status.calibration_in_progress:
        raise ValueError(f'Robot is not ready for a tool reset. Arm errors: {arm_errors}')
    print('Only the gripper motor will reboot. No arm movement commands are sent.')
    print('The gripper must be EMPTY and free of contact. Keep hands away from the fingers.')
    if confirm('Type RESET EMPTY GRIPPER to reset, anything else cancels: ').strip() != 'RESET EMPTY GRIPPER':
        print('Cancelled. No reset performed.')
        return False
    robot.tool_reboot()
    deadline = time.monotonic() + 5
    while True:
        status = robot.get_hardware_status()
        errors = [(name, code) for name, code in zip(status.motor_names, status.hardware_errors) if code != 0]
        if not errors and status.connection_up:
            print('Gripper reset completed. No motor errors reported.')
            return True
        if time.monotonic() >= deadline:
            raise ValueError(f'Errors remain after reset: {errors}. Details: {status.error_message}')
        time.sleep(.2)


def main():
    robot = None
    try:
        robot = NiryoRobot(ROBOT_IP)
        reset_gripper(robot)
    except Exception as error:
        print(f'Reset stopped: {type(error).__name__}: {error}')
    finally:
        if robot is not None:
            robot.close_connection()


if __name__ == '__main__':
    main()
