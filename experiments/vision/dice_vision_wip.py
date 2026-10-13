from pyniryo import NiryoRobot
import numpy as np
import cv2

ROBOT_IP = "169.254.200.200"

robot = None

try:
    robot = NiryoRobot(ROBOT_IP)

    print("ROBOT CONNECTED")
    print("Dice detection + rotation-independent dot counting started.")
    print("Press Q to quit.")

    while True:

        # ==================================================
        # GET CAMERA IMAGE
        # ==================================================

        img_comp = robot.get_img_compressed()

        img_array = np.frombuffer(
            img_comp,
            dtype=np.uint8
        )

        frame = cv2.imdecode(
            img_array,
            cv2.IMREAD_COLOR
        )

        if frame is None:
            continue

        # ==================================================
        # GRAYSCALE
        # ==================================================

        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY
        )

        # ==================================================
        # DICE OBJECT DETECTION
        # ==================================================

        # Black dice -> white
        _, binary = cv2.threshold(
            gray,
            80,
            255,
            cv2.THRESH_BINARY_INV
        )

        # Fill white dots
        kernel = np.ones(
            (15, 15),
            np.uint8
        )

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
        total_dots = 0

        # Full-frame mask for debugging
        dot_mask_display = np.zeros_like(gray)

        # ==================================================
        # PROCESS EACH DIE
        # ==================================================

        for contour in contours:

            area = cv2.contourArea(contour)

            # Ignore small objects
            if area < 1000:
                continue

            # Bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)

            # Ignore very large objects
            if w > 400 or h > 400:
                continue

            # Dice should approximately be square
            aspect_ratio = w / float(h)

            if not (0.6 <= aspect_ratio <= 1.4):
                continue

            dice_count += 1

            # ==================================================
            # CREATE MASK OF ACTUAL DIE SHAPE
            # ==================================================

            die_mask = np.zeros_like(gray)

            cv2.drawContours(
                die_mask,
                [contour],
                -1,
                255,
                -1
            )

            # ==================================================
            # SHRINK DIE MASK
            # ==================================================

            # This removes the outer edge of the die.
            # It also removes background from the corners
            # of the bounding rectangle.

            erosion_kernel = np.ones(
                (9, 9),
                np.uint8
            )

            inner_die_mask = cv2.erode(
                die_mask,
                erosion_kernel,
                iterations=1
            )

            # ==================================================
            # ORIGINAL GRAYSCALE INSIDE DIE
            # ==================================================

            die_gray = gray.copy()

            # ==================================================
            # FIND WHITE PIPS
            # ==================================================

            _, white_pixels = cv2.threshold(
                die_gray,
                160,
                255,
                cv2.THRESH_BINARY
            )

            # Only keep white pixels INSIDE the actual die
            dot_mask = cv2.bitwise_and(
                white_pixels,
                inner_die_mask
            )

            # Small cleanup
            dot_kernel = np.ones(
                (3, 3),
                np.uint8
            )

            dot_mask = cv2.morphologyEx(
                dot_mask,
                cv2.MORPH_OPEN,
                dot_kernel
            )

            # Add to debug image
            dot_mask_display = cv2.bitwise_or(
                dot_mask_display,
                dot_mask
            )

            # ==================================================
            # FIND WHITE COMPONENTS
            # ==================================================

            num_labels, labels, stats, centroids = (
                cv2.connectedComponentsWithStats(
                    dot_mask,
                    connectivity=8
                )
            )

            dot_count = 0

            # ==================================================
            # CHECK COMPONENTS
            # ==================================================

            for i in range(1, num_labels):

                dot_area = stats[
                    i,
                    cv2.CC_STAT_AREA
                ]

                dot_w = stats[
                    i,
                    cv2.CC_STAT_WIDTH
                ]

                dot_h = stats[
                    i,
                    cv2.CC_STAT_HEIGHT
                ]

                # ------------------------------------------
                # SIZE
                # ------------------------------------------

                if dot_area < 20:
                    continue

                if dot_area > 2000:
                    continue

                # ------------------------------------------
                # ASPECT RATIO
                # ------------------------------------------

                dot_ratio = dot_w / float(dot_h)

                if not (
                    0.5 <= dot_ratio <= 2.0
                ):
                    continue

                # ------------------------------------------
                # CONTOUR
                # ------------------------------------------

                component = (
                    np.uint8(labels == i) * 255
                )

                dot_contours, _ = cv2.findContours(
                    component,
                    cv2.RETR_EXTERNAL,
                    cv2.CHAIN_APPROX_SIMPLE
                )

                if not dot_contours:
                    continue

                dot_contour = max(
                    dot_contours,
                    key=cv2.contourArea
                )

                # ------------------------------------------
                # CIRCULARITY
                # ------------------------------------------

                perimeter = cv2.arcLength(
                    dot_contour,
                    True
                )

                if perimeter == 0:
                    continue

                circularity = (
                    4.0 * np.pi * dot_area
                    / (perimeter * perimeter)
                )

                if circularity < 0.40:
                    continue

                # ==================================================
                # VALID PIP
                # ==================================================

                dot_count += 1

                cx = int(
                    centroids[i][0]
                )

                cy = int(
                    centroids[i][1]
                )

                # Draw detected pip
                cv2.circle(
                    detection,
                    (cx, cy),
                    6,
                    (255, 0, 0),
                    -1
                )

                cv2.circle(
                    detection,
                    (cx, cy),
                    int(max(dot_w, dot_h) / 2),
                    (255, 0, 0),
                    2
                )

            # ==================================================
            # TOTAL
            # ==================================================

            total_dots += dot_count

            # ==================================================
            # DIE CENTER
            # ==================================================

            center_x = x + w // 2
            center_y = y + h // 2

            # ==================================================
            # DRAW BOUNDING BOX
            # ==================================================

            cv2.rectangle(
                detection,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                3
            )

            # ==================================================
            # DRAW ACTUAL DIE CONTOUR
            # ==================================================

            cv2.drawContours(
                detection,
                [contour],
                -1,
                (0, 255, 255),
                2
            )

            # ==================================================
            # DRAW CENTER
            # ==================================================

            cv2.circle(
                detection,
                (center_x, center_y),
                5,
                (0, 0, 255),
                -1
            )

            # ==================================================
            # LABEL
            # ==================================================

            cv2.putText(
                detection,
                f"Die {dice_count}: {dot_count}",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

        # ==================================================
        # TOTAL DICE
        # ==================================================

        cv2.putText(
            detection,
            f"Dice: {dice_count}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2
        )

        # ==================================================
        # TOTAL DOTS
        # ==================================================

        cv2.putText(
            detection,
            f"SUM: {total_dots}",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2
        )

        # ==================================================
        # DISPLAY
        # ==================================================

        cv2.imshow(
            "Dice Detection",
            detection
        )

        cv2.imshow(
            "Binary - Dice",
            binary
        )

        cv2.imshow(
            "Filled Dice",
            binary_filled
        )

        cv2.imshow(
            "Dot Mask",
            dot_mask_display
        )

        # ==================================================
        # QUIT
        # ==================================================

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