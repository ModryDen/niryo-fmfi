from pyniryo import NiryoRobot
import numpy as np
import cv2

ROBOT_IP = "169.254.200.200"

# ROI: left, top, right, bottom
ROI = (0.33, 0.10, 0.68, 0.58)

robot = None

try:
    robot = NiryoRobot(ROBOT_IP)

    print("ROBOT CONNECTED")
    print("Dice detection started.")
    print("Press Q to quit.")

    while True:

        # Get image
        img_comp = robot.get_img_compressed()

        img_array = np.frombuffer(img_comp, dtype=np.uint8)
        frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        if frame is None:
            continue

        frame_height, frame_width = frame.shape[:2]

        roi_x0 = int(frame_width * ROI[0])
        roi_y0 = int(frame_height * ROI[1])
        roi_x1 = int(frame_width * ROI[2])
        roi_y1 = int(frame_height * ROI[3])

        roi = frame[roi_y0:roi_y1, roi_x0:roi_x1]

        # Convert to grayscale
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        # Threshold: black dice become white objects
        _, binary = cv2.threshold(
            gray,
            140,
            255,
            cv2.THRESH_BINARY_INV
        )

        # Fill white pips inside dice
        kernel = np.ones((7, 7), np.uint8)

        binary_filled = cv2.morphologyEx(
            binary,
            cv2.MORPH_CLOSE,
            kernel
        )

        contours, _ = cv2.findContours(
            binary_filled,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        detection = frame.copy()

        cv2.rectangle(
            detection,
            (roi_x0, roi_y0),
            (roi_x1, roi_y1),
            (255, 255, 0),
            2
        )

        dice_count = 0

        for contour in contours:

            area = cv2.contourArea(contour)

            # Ignore tiny noise
            if area < 150:
                continue

            x, y, w, h = cv2.boundingRect(contour)

            # Ignore giant objects
            if w > 150 or h > 150:
                continue

            aspect_ratio = w / float(h)

            # Dice should be roughly square
            if not (0.7 <= aspect_ratio <= 1.3):
                continue

            # Check contour shape
            perimeter = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(
                contour,
                0.04 * perimeter,
                True
            )

            # Dice should appear as quadrilaterals
            if len(approx) < 4:
                continue

            dice_count += 1

            # Convert ROI coordinates to full image coordinates
            x_global = x + roi_x0
            y_global = y + roi_y0

            cv2.rectangle(
                detection,
                (x_global, y_global),
                (x_global + w, y_global + h),
                (0, 255, 0),
                2
            )

            center_x = x_global + w // 2
            center_y = y_global + h // 2

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
                (x_global, y_global - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

            print(
                f"Die {dice_count}: "
                f"area={area:.0f} "
                f"size={w}x{h}"
            )

        cv2.putText(
            detection,
            f"Dice: {dice_count}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2
        )

        cv2.imshow("Dice Detection", detection)
        cv2.imshow("Binary", binary)
        cv2.imshow("Filled Dice", binary_filled)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

except Exception as e:

    print("DETECTION FAILED")
    print(e)

finally:

    cv2.destroyAllWindows()

    if robot is not None:
        robot.close_connection()

    print("Connection closed.")
