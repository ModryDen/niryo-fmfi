from pyniryo import NiryoRobot

ROBOT_IP = "169.254.200.200"

print("Connecting to robot...")

try:
    robot = NiryoRobot(ROBOT_IP)
    print("ROBOT CONNECTED")

    robot.close_connection()
    print("Connection closed.")

except Exception as e:
    print("ROBOT NOT CONNECTED")
    print(f"Error: {e}")