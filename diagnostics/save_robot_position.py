from pyniryo import NiryoRobot
import json

ROBOT_IP = "169.254.200.200"

robot = None

try:
    robot = NiryoRobot(ROBOT_IP)
    print("ROBOT CONNECTED")

    joints = robot.get_joints()

    print("Current joints:")
    print(joints)

    with open("camera_position.json", "w") as f:
        json.dump(joints, f, indent=4)

    print("Position saved to camera_position.json")

except Exception as e:
    print("SAVE FAILED")
    print(f"Error: {e}")

finally:
    if robot is not None:
        robot.close_connection()

    print("Connection closed.")
