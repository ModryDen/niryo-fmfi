from pyniryo import NiryoRobot
import time

ROBOT_IP = "169.254.200.200"

robot = None

try:
    print("Connecting to robot...")
    robot = NiryoRobot(ROBOT_IP)
    print("ROBOT CONNECTED")
    print("Press Ctrl+C to stop.")

    joints = robot.get_joints()

    while True:
        joints[5] += 1.0
        robot.move_joints(*joints)
        time.sleep(0.5)

        joints[5] -= 1.0
        robot.move_joints(*joints)
        time.sleep(0.5)

except KeyboardInterrupt:
    print("Movement stopped.")

except Exception as e:
    print("MOVEMENT FAILED")
    print(f"Error: {e}")

finally:
    if robot is not None:
        robot.close_connection()
        print("Connection closed.")