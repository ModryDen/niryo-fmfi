from pyniryo import NiryoRobot
import numpy as np
import cv2

ROBOT_IP = "169.254.200.200"

robot = None

try:
    robot = NiryoRobot(ROBOT_IP)
    print("ROBOT CONNECTED")

    img_comp = robot.get_img_compressed()

    # Decode JPEG/PNG bytes directly with modern NumPy/OpenCV
    img_array = np.frombuffer(img_comp, dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    if img is None:
        raise RuntimeError("Could not decode camera image")

    cv2.imwrite("ned_frame.png", img)

    print("CAMERA IMAGE CAPTURED")
    print(f"Image size: {img.shape[1]} x {img.shape[0]}")

    cv2.imshow("NED camera", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

except Exception as e:
    print("CAMERA TEST FAILED")
    print(f"Error: {e}")

finally:
    if robot is not None:
        robot.close_connection()
        print("Connection closed.")