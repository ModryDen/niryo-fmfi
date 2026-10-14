from pyniryo import NiryoRobot

ROBOT_IP = "169.254.200.200"

pos = [
    -0.021666156231653747,
    0.5462243577263421,
    -0.6891070714947368,
    0.015432461468649183,
    -1.4466365365659528,
    -0.12472458503935746
]

robot = None

try:
    robot = NiryoRobot(ROBOT_IP)
    print("ROBOT CONNECTED")

    robot.move_joints(pos)

    print("Robot moved to the saved position.")

except Exception as e:
    print("MOVE FAILED")
    print(f"Error: {e}")

finally:
    if robot is not None:
        robot.close_connection()

    print("Connection closed.")