from pyniryo import NiryoRobot
import numpy as np
import cv2

ROBOT_IP = "169.254.200.200"

robot = None

try:
    robot = NiryoRobot(ROBOT_IP)

    print("ROBOT CONNECTED")
    print("Dice detection started.")
    print("Press Q to quit.")

    while True:

        # Get camera image
        img_comp = robot.get_img_compressed()

        img_array = np.frombuffer(img_comp, dtype=np.uint8)
        frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        if frame is None:
            continue

        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Black dice -> white
        _, binary = cv2.threshold(
            gray,
            130,
            255,
            cv2.THRESH_BINARY_INV
        )

        # Fill the white dots inside the black dice
        kernel = np.ones((15, 15), np.uint8)

        binary_filled = cv2.morphologyEx(
            binary,
            cv2.MORPH_CLOSE,
            kernel
        )

        # Find objects
        contours, _ = cv2.findContours(
            binary_filled,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        detection = frame.copy()
        dice_count = 0

        for contour in contours:

            area = cv2.contourArea(contour)

            # Ignore small objects
            if area < 1000:
                continue

            x, y, w, h = cv2.boundingRect(contour)

            # Ignore very large objects
            if w > 400 or h > 400:
                continue

            # Dice should be approximately square
            aspect_ratio = w / float(h)

            if 0.6 <= aspect_ratio <= 1.4:

                dice_count += 1

                # Bounding box
                cv2.rectangle(
                    detection,
                    (x, y),
                    (x + w, y + h),
                    (0, 255, 0),
                    3
                )

                # Center
                center_x = x + w // 2
                center_y = y + h // 2

                cv2.circle(
                    detection,
                    (center_x, center_y),
                    5,
                    (0, 0, 255),
                    -1
                )

                cv2.putText(
                    detection,
                    f"Die {dice_count}",
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )

        # Display count
        cv2.putText(
            detection,
            f"Dice: {dice_count}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2
        )

        # Show results
        cv2.imshow("Dice Detection", detection)
        cv2.imshow("Binary - Dice", binary)
        cv2.imshow("Filled Dice", binary_filled)

        # Q = quit
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

except Exception as e:

    print("DETECTION FAILED")
    print(f"Error: {e}")

finally:

    cv2.destroyAllWindows()

    if robot is not None:
        robot.close_connection()

    print("Connection closed.")