
from pyniryo import NiryoRobot
import numpy as np
import cv2
import time
from collections import deque

ROBOT_IP = "169.254.200.200"
FPS_WINDOW = 1.0  # seconds

robot = None
frame_times = deque()

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

        now = time.perf_counter()
        frame_times.append(now)

        # Keep only timestamps from the last second
        while frame_times and now - frame_times[0] > FPS_WINDOW:
            frame_times.popleft()

        fps = len(frame_times)

        cv2.putText(
            frame,
            f"FPS: {fps}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        cv2.imshow("NED Camera FPS", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

except Exception as e:
    print("CAMERA FPS TEST FAILED")
    print(f"Error: {e}")

finally:
    cv2.destroyAllWindows()

    if robot is not None:
        robot.close_connection()
        print("Connection closed.")
