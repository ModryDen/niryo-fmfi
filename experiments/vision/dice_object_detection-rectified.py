from pathlib import Path
import argparse

from pyniryo import NiryoRobot
import numpy as np
import cv2

from dice_camera_calibration import (
    CALIBRATION_FILE,
    confirmed_move,
    load_calibration,
    read_frame,
    require_pose,
    teach_calibration,
)

ROBOT_IP = "169.254.200.200"

SAVE_DIR = Path("dice_dataset")

# Normalized ROI: left, top, right, bottom
ROI = (0.33, 0.10, 0.68, 0.58)

MIN_AREA = 150
MAX_DIE_SIZE = 150
MIN_ASPECT_RATIO = 0.65

WARP_SIZE = 256
CLOSE_KERNEL_SIZE = 7
OPEN_KERNEL_SIZE = 3

ECC_ITERATIONS = 100
ECC_EPSILON = 1e-6

reference_gray = None


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

    warped = cv2.warpPerspective(image, matrix, (size, size))
    warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)

    _, warped_binary = cv2.threshold(
        warped_gray,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    return warped, warped_gray, warped_binary, rect, box


def align_ecc(reference, image):
    """Align image to reference using an affine ECC transform."""
    if reference is None:
        return image, None

    reference_float = reference.astype(np.float32) / 255.0
    image_float = image.astype(np.float32) / 255.0

    warp_matrix = np.eye(2, 3, dtype=np.float32)

    criteria = (
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
        ECC_ITERATIONS,
        ECC_EPSILON
    )

    try:
        score, warp_matrix = cv2.findTransformECC(
            reference_float,
            image_float,
            warp_matrix,
            cv2.MOTION_AFFINE,
            criteria
        )

        aligned = cv2.warpAffine(
            image,
            warp_matrix,
            (reference.shape[1], reference.shape[0]),
            flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=255
        )

        _, aligned_binary = cv2.threshold(
            aligned,
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        return aligned_binary, score

    except cv2.error:
        # ECC can fail when images are too different.
        return image, None


def detect_dice(frame):
    height, width = frame.shape[:2]

    x0 = int(width * ROI[0])
    y0 = int(height * ROI[1])
    x1 = int(width * ROI[2])
    y1 = int(height * ROI[3])

    roi = frame[y0:y1, x0:x1]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    _, binary = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    open_kernel = np.ones(
        (OPEN_KERNEL_SIZE, OPEN_KERNEL_SIZE), np.uint8
    )
    close_kernel = np.ones(
        (CLOSE_KERNEL_SIZE, CLOSE_KERNEL_SIZE), np.uint8
    )

    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, open_kernel)
    binary_filled = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, close_kernel)

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

        if hull_area == 0 or area / hull_area < 0.80:
            continue

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

        warped, warped_gray, warped_binary, _, _ = rectify_die(
            frame, global_contour
        )

        detections.append({
            "center": center,
            "angle": angle,
            "area": area,
            "size": (rw, rh),
            "contour": global_contour,
            "box": box,
            "warped": warped,
            "warped_gray": warped_gray,
            "warped_binary": warped_binary,
        })

    detections.sort(
        key=lambda die: (die["center"][1], die["center"][0])
    )

    return detections, (x0, y0, x1, y1)


def draw_detections(frame, detections, roi_bounds):
    output = frame.copy()
    x0, y0, x1, y1 = roi_bounds

    cv2.rectangle(output, (x0, y0), (x1, y1), (255, 255, 0), 2)

    for index, die in enumerate(detections, start=1):
        cv2.drawContours(output, [die["contour"]], -1, (0, 255, 0), 2)
        cv2.polylines(output, [die["box"]], True, (255, 0, 255), 2)

        cx, cy = die["center"]
        cv2.circle(output, (cx, cy), 4, (0, 0, 255), -1)
        cv2.putText(
            output, f"Die {index}", (cx + 8, cy - 8),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
        )

    cv2.putText(
        output, f"Dice: {len(detections)}", (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2
    )

    return output


def main():
    global reference_gray

    parser = argparse.ArgumentParser(description="Detect dice and teach camera close-up poses.")
    parser.add_argument("--calibrate", action="store_true",
                        help="Teach camera mapping manually; does not move the robot.")
    args = parser.parse_args()
    robot = None
    open_die_windows = set()
    selection = {"pixel": None}
    close_up_pose = None

    def select_die(event, x, y, _flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN:
            selection["pixel"] = (x, y)

    try:
        robot = NiryoRobot(ROBOT_IP)
        print("ROBOT CONNECTED")
        intrinsics = robot.get_camera_intrinsics()
        if args.calibrate:
            teach_calibration(robot, intrinsics)
            return
        calibration = load_calibration(intrinsics) if CALIBRATION_FILE.exists() else None
        if calibration is None:
            print("Motion disabled: first run this script with --calibrate.")
        print("Click a die | M = confirm close-up | B = confirm return to overview")
        print("Q = quit (no automatic return) | R = reset alignment reference | S = save")
        cv2.namedWindow("Dice Detection")
        cv2.setMouseCallback("Dice Detection", select_die)

        while True:
            frame = read_frame(robot, intrinsics)

            if close_up_pose is not None:
                cv2.imshow("Dice Detection", frame)
                key = cv2.waitKey(20) & 0xFF
                if key == ord("q"):
                    break
                if key == ord("b"):
                    raised = close_up_pose.copy()
                    raised[2] = calibration.overview[2]
                    if confirmed_move(
                        robot, close_up_pose, [raised, calibration.overview]
                    ):
                        close_up_pose = None
                        selection["pixel"] = None
                continue

            detections, roi_bounds = detect_dice(frame)
            selected = None
            if selection["pixel"] is not None:
                for die in detections:
                    if cv2.pointPolygonTest(
                        die["contour"], selection["pixel"], False
                    ) >= 0:
                        selected = die
                        break
            output = draw_detections(frame, detections, roi_bounds)
            if selected is not None:
                cv2.drawContours(output, [selected["contour"]], -1, (0, 255, 255), 3)
                cv2.imshow("Selected Die", selected["warped"])

            cv2.imshow("Dice Detection", output)

            current_windows = set()

            for index, die in enumerate(detections, start=1):
                window_name = f"Die {index} - Binary"

                if reference_gray is None:
                    reference_gray = die["warped_gray"].copy()

                aligned_binary, score = align_ecc(
                    reference_gray,
                    die["warped_gray"]
                )

                cv2.imshow(window_name, aligned_binary)
                current_windows.add(window_name)

            for window_name in open_die_windows - current_windows:
                cv2.destroyWindow(window_name)

            open_die_windows = current_windows

            key = cv2.waitKey(1) & 0xFF

            if key == ord("s"):
                SAVE_DIR.mkdir(parents=True, exist_ok=True)

                for index, die in enumerate(detections, start=1):
                    filename = SAVE_DIR / f"die_{index}_{cv2.getTickCount()}.png"
                    cv2.imwrite(str(filename), die["warped_binary"])
                    print(f"Saved: {filename}")

            if key == ord("q"):
                break
            elif key == ord("r"):
                reference_gray = None
                print("Alignment reference reset.")
            elif key == ord("m"):
                if calibration is None:
                    print("Motion blocked: calibration is missing. Run with --calibrate.")
                    continue
                if selected is None:
                    print("Motion blocked: click inside a currently detected die.")
                    continue
                try:
                    require_pose(robot.get_pose().to_list(), calibration.overview)
                    target = calibration.target(
                        selected["center"], (frame.shape[1], frame.shape[0])
                    )
                except ValueError as error:
                    print(f"Motion blocked: {error}")
                    continue
                raised = target.copy()
                raised[2] = calibration.overview[2]
                print(f"Selected die centre (undistorted pixels): {selected['center']}")
                if confirmed_move(robot, calibration.overview, [raised, target]):
                    close_up_pose = target
                    for window_name in open_die_windows:
                        cv2.destroyWindow(window_name)
                    open_die_windows.clear()
                    cv2.destroyWindow("Selected Die")
                    print("Close-up view. B = confirmed return; Q leaves the robot here.")

    except Exception as e:
        print("DETECTION FAILED")
        print(f"{type(e).__name__}: {e}")

    finally:
        cv2.destroyAllWindows()

        if robot is not None:
            robot.close_connection()

        print("Connection closed.")


if __name__ == "__main__":
    main()
