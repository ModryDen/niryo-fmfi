
from pyniryo import NiryoRobot
import numpy as np
import cv2

ROBOT_IP = "169.254.200.200"

# Normalized ROI: left, top, right, bottom
ROI = (0.33, 0.10, 0.68, 0.58)

MIN_AREA = 150
MAX_DIE_SIZE = 150
MIN_ASPECT_RATIO = 0.65

WARP_SIZE = 256
CLOSE_KERNEL_SIZE = 7
OPEN_KERNEL_SIZE = 3


def order_corners(points):
    """Order four points as top-left, top-right, bottom-right, bottom-left."""
    points = np.asarray(points, dtype=np.float32).reshape(4, 2)

    center = points.mean(axis=0)
    angles = np.arctan2(
        points[:, 1] - center[1],
        points[:, 0] - center[0]
    )

    points = points[np.argsort(angles)]
    start = np.argmin(points[:, 0] + points[:, 1])

    return np.roll(points, -start, axis=0)


def rectify_die(image, contour):
    """Rectify a detected die and return its color and binary images."""
    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)
    source = order_corners(box)

    size = WARP_SIZE
    destination = np.array(
        [
            [0, 0],
            [size - 1, 0],
            [size - 1, size - 1],
            [0, size - 1]
        ],
        dtype=np.float32
    )

    matrix = cv2.getPerspectiveTransform(source, destination)

    warped = cv2.warpPerspective(
        image,
        matrix,
        (size, size)
    )

    # Convert rectified image to grayscale
    warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)

    # Binarize using Otsu thresholding
    threshold_value, warped_binary = cv2.threshold(
        warped_gray,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    return warped, warped_binary, rect, box, threshold_value


def detect_dice(frame):
    height, width = frame.shape[:2]

    x0 = int(width * ROI[0])
    y0 = int(height * ROI[1])
    x1 = int(width * ROI[2])
    y1 = int(height * ROI[3])

    roi = frame[y0:y1, x0:x1]

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # Binarize the ROI using Otsu thresholding
    _, binary = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    open_kernel = np.ones(
        (OPEN_KERNEL_SIZE, OPEN_KERNEL_SIZE),
        np.uint8
    )

    close_kernel = np.ones(
        (CLOSE_KERNEL_SIZE, CLOSE_KERNEL_SIZE),
        np.uint8
    )

    # Remove small noise
    binary = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        open_kernel
    )

    # Fill small holes and gaps
    binary_filled = cv2.morphologyEx(
        binary,
        cv2.MORPH_CLOSE,
        close_kernel
    )

    contours, _ = cv2.findContours(
        binary_filled,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    detections = []

    for contour in contours:
        area = cv2.contourArea(contour)

        if area < MIN_AREA:
            continue

        rect = cv2.minAreaRect(contour)
        (cx, cy), (rw, rh), angle = rect

        if rw <= 0 or rh <= 0:
            continue

        short_side = min(rw, rh)
        long_side = max(rw, rh)

        if long_side > MAX_DIE_SIZE:
            continue

        if short_side / long_side < MIN_ASPECT_RATIO:
            continue

        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)

        if hull_area == 0:
            continue

        if area / hull_area < 0.80:
            continue

        # Convert contour coordinates from ROI to full image
        global_contour = contour.copy()
        global_contour[:, 0, 0] += x0
        global_contour[:, 0, 1] += y0

        center = (
            int(round(cx + x0)),
            int(round(cy + y0))
        )

        box = cv2.boxPoints(rect)
        box[:, 0] += x0
        box[:, 1] += y0
        box = np.int32(np.round(box))

        # Rectify the die, then binarize the rectified image
        warped, warped_binary, _, _, threshold_value = rectify_die(
            frame,
            global_contour
        )

        detections.append({
            "center": center,
            "angle": angle,
            "area": area,
            "size": (rw, rh),
            "contour": global_contour,
            "box": box,
            "warped": warped,
            "warped_binary": warped_binary,
            "warp_threshold": threshold_value
        })

    detections.sort(
        key=lambda die: (
            die["center"][1],
            die["center"][0]
        )
    )

    return detections, binary, binary_filled, (x0, y0, x1, y1)


def draw_detections(frame, detections, roi_bounds):
    output = frame.copy()

    x0, y0, x1, y1 = roi_bounds

    cv2.rectangle(
        output,
        (x0, y0),
        (x1, y1),
        (255, 255, 0),
        2
    )

    for index, die in enumerate(detections, start=1):
        cv2.drawContours(
            output,
            [die["contour"]],
            -1,
            (0, 255, 0),
            2
        )

        cv2.polylines(
            output,
            [die["box"]],
            True,
            (255, 0, 255),
            2
        )

        cx, cy = die["center"]

        cv2.circle(
            output,
            (cx, cy),
            4,
            (0, 0, 255),
            -1
        )

        cv2.putText(
            output,
            f"Die {index}",
            (cx + 8, cy - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2
        )

    cv2.putText(
        output,
        f"Dice: {len(detections)}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 255),
        2
    )

    return output


def main():
    robot = None

    try:
        robot = NiryoRobot(ROBOT_IP)

        print("ROBOT CONNECTED")
        print("Dice detection started.")
        print("Press Q to quit.")

        while True:
            img_comp = robot.get_img_compressed()
            img_array = np.frombuffer(img_comp, dtype=np.uint8)

            frame = cv2.imdecode(
                img_array,
                cv2.IMREAD_COLOR
            )

            if frame is None:
                continue

            detections, binary, filled, roi_bounds = detect_dice(frame)

            output = draw_detections(
                frame,
                detections,
                roi_bounds
            )

            cv2.imshow("Dice Detection", output)
            cv2.imshow("Binary - Otsu", binary)
            cv2.imshow("Filled Dice", filled)

            if detections:
                # Show the first detected die, rectified and binarized
                cv2.imshow(
                    "Rectified Die - Binary",
                    detections[0]["warped_binary"]
                )

                # Optional: inspect the rectified color image
                cv2.imshow(
                    "Rectified Die - Color",
                    detections[0]["warped"]
                )
            else:
                empty = np.zeros(
                    (WARP_SIZE, WARP_SIZE),
                    dtype=np.uint8
                )

                cv2.imshow("Rectified Die - Binary", empty)
                cv2.imshow(
                    "Rectified Die - Color",
                    np.zeros(
                        (WARP_SIZE, WARP_SIZE, 3),
                        dtype=np.uint8
                    )
                )

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


if __name__ == "__main__":
    main()
