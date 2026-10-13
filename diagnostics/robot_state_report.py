from pyniryo import NiryoRobot

ROBOT_IP = "169.254.200.200"

robot = None

try:
    robot = NiryoRobot(ROBOT_IP)

    print("ROBOT CONNECTED")
    print("\nCURRENT JOINTS:")
    print(robot.get_joints())

    print("\nCURRENT POSE:")
    print(robot.get_pose())

finally:
    if robot is not None:
        robot.close_connection()