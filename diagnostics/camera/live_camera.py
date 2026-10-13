from pyniryo import NiryoRobot
import numpy as np
import cv2

ROBOT_IP = "169.254.200.200"

robot = None

try:
    robot = NiryoRobot(ROBOT_IP)
    print("ROBOT CONNECTED")
    print("Press Q to quit.")

    while True:
        img_comp = robot.get_img_compressed()

        img_array = np.frombuffer(img_comp, dtype=np.uint8)
        frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        if frame is None:
            continue

        cv2.imshow("NED Camera", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

except Exception as e:
    print("CAMERA TEST FAILED")
    print(f"Error: {e}")

finally:
    cv2.destroyAllWindows()

    if robot is not None:
        robot.close_connection()
        print("Connection closed.")