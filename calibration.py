from pyniryo import NiryoRobot

ROBOT_IP = "169.254.200.200"
print("Connecting to robot...")
robot = None

try:
    robot = NiryoRobot(ROBOT_IP)
    print("ROBOT CONNECTED")
    print("Starting automatic calibration...")
    robot.calibrate_auto()
    print("CALIBRATION SUCCESSFUL")

except Exception as e:
    print("CALIBRATION FAILED")
    print(f"Error: {e}")

finally:
    if robot is not None:
        robot.close_connection()
        print("Connection closed.")