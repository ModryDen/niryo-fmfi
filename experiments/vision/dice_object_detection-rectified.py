from pathlib import Path
import argparse
from dataclasses import dataclass

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

MIN_AREA = 150
MAX_DIE_SIZE = 150
MIN_ASPECT_RATIO = 0.65

WARP_SIZE = 256
CLOSE_KERNEL_SIZE = 7
OPEN_KERNEL_SIZE = 3

ECC_ITERATIONS = 100
ECC_EPSILON = 1e-6

reference_gray = None


@dataclass
class Workspace:
    corners: np.ndarray
    mask: np.ndarray

    def relative_position(self, pixel):
        destination = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float32)
        transform = cv2.getPerspectiveTransform(
            self.corners.astype(np.float32), destination
        )
        point = np.asarray([[pixel]], dtype=np.float32)
        relative = cv2.perspectiveTransform(point, transform)[0, 0]
        return float(relative[0]), float(relative[1])


def find_ring_markers(gray):
    adaptive = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, 25
    )
    low_contrast = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 5
    )
    _, global_threshold = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    candidates = []
    for threshold in (adaptive, low_contrast, global_threshold):
        contours, hierarchy = cv2.findContours(
            threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )
        if hierarchy is None:
            continue
        for index, contour in enumerate(contours):
            (cx, cy), radius = cv2.minEnclosingCircle(contour)
            if not 7 <= radius <= min(gray.shape) * 0.12:
                continue
            center = np.array([cx, cy])
            nested = 0
            current = index
            while current >= 0:
                ring = contours[current]
                area = cv2.contourArea(ring)
                perimeter = cv2.arcLength(ring, True)
                if perimeter == 0:
                    break
                circularity = 4 * np.pi * area / perimeter ** 2
                if circularity < (0.83 if nested == 0 else 0.70):
                    break
                (rx, ry), _ = cv2.minEnclosingCircle(ring)
                if np.linalg.norm(np.array([rx, ry]) - center) > max(2, radius * 0.15):
                    break
                nested += 1
                current = hierarchy[0, current, 2]
            if nested >= 3:
                candidates.append((center, radius))

    # Each ring and threshold can produce the same marker; keep its largest contour.
    markers = []
    for center, radius in sorted(candidates, key=lambda item: item[1], reverse=True):
        if not any(
            np.linalg.norm(center - other) <= max(3, other_radius * 0.25)
            for other, other_radius in markers
        ):
            markers.append((center, radius))
    return markers


def detect_workspace(frame):
    """Locate exactly four Niryo ring markers without guessing missing corners."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    markers = find_ring_markers(gray)
    if len(markers) != 4:
        return None

    centers = np.asarray([center for center, _ in markers], dtype=np.float32)
    hull = cv2.convexHull(centers).reshape(-1, 2)
    if len(hull) != 4 or cv2.contourArea(hull) < 2500:
        return None
    corners = np.int32(np.round(order_corners(hull)))
    sides = np.linalg.norm(np.roll(corners, -1, axis=0) - corners, axis=1)
    radii = [radius for _, radius in markers]
    if min(sides) < 4 * max(radii) or max(sides) / min(sides) > 4:
        return None

    mask = np.zeros(gray.shape, dtype=np.uint8)
    cv2.fillConvexPoly(mask, corners, 255)
    mask = cv2.erode(mask, np.ones((5, 5), np.uint8))
    for center, radius in markers:
        point = tuple(map(int, np.round(center)))
        cv2.circle(mask, point, int(np.ceil(radius)) + 3, 0, -1)
    return Workspace(corners, mask)


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
    workspace = detect_workspace(frame)
    if workspace is None:
        return [], None

    x0, y0, width, height = cv2.boundingRect(workspace.corners)
    x1, y1 = x0 + width, y0 + height
    roi = frame[y0:y1, x0:x1]
    workspace_mask = workspace.mask[y0:y1, x0:x1]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    threshold, _ = cv2.threshold(
        gray[workspace_mask != 0], 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    _, binary = cv2.threshold(
        gray, threshold, 255, cv2.THRESH_BINARY_INV
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

        contour_mask = np.zeros(gray.shape, dtype=np.uint8)
        cv2.drawContours(contour_mask, [contour], -1, 255, -1)
        if np.any((contour_mask != 0) & (workspace_mask == 0)):
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
            "workspace_position": workspace.relative_position(center),
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

    return detections, workspace


def draw_detections(frame, detections, workspace):
    output = frame.copy()

    if workspace is None:
        cv2.putText(
            output, "Workspace NOT found: show all 4 ring markers",
            (10, frame.shape[0] - 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2
        )
    else:
        contours, _ = cv2.findContours(
            workspace.mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(output, contours, -1, (255, 255, 0), 2)
        for index, corner in enumerate(workspace.corners, start=1):
            point = tuple(map(int, corner))
            cv2.circle(output, point, 5, (0, 255, 255), -1)
            cv2.putText(
                output, f"M{index}", (point[0] + 8, point[1] - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2
            )

    for index, die in enumerate(detections, start=1):
        cv2.drawContours(output, [die["contour"]], -1, (0, 255, 0), 2)
        cv2.polylines(output, [die["box"]], True, (255, 0, 255), 2)

        cx, cy = die["center"]
        cv2.circle(output, (cx, cy), 4, (0, 0, 255), -1)
        cv2.putText(
            output, f"Die {index}", (cx + 8, cy - 8),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
        )
        x_rel, y_rel = die["workspace_position"]
        cv2.putText(
            output, f"x={x_rel:.2f} y={y_rel:.2f}", (cx + 8, cy + 14),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1
        )

    cv2.putText(
        output, f"Dice: {len(detections)}" if workspace is not None else "Dice: --",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2
    )

    return output


def main():
    global reference_gray

    parser = argparse.ArgumentParser(description="Detect dice and teach camera close-up poses.")
    parser.add_argument("--calibrate", action="store_true",
                        help="Teach camera mapping manually; does not move the robot.")
    parser.add_argument("--enable-motion", action="store_true",
                        help="Allow confirmed moves with a taught camera calibration. "
                             "Default: read-only dice positions, no motion.")
    args = parser.parse_args()
    robot = None
    open_die_windows = set()
    selection = {"pixel": None, "pending": False}
    selected_window_open = False
    close_up_pose = None
    workspace_visible = None

    def select_die(event, x, y, _flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN:
            selection["pixel"] = (x, y)
            selection["pending"] = True

    try:
        robot = NiryoRobot(ROBOT_IP)
        print("ROBOT CONNECTED")
        intrinsics = robot.get_camera_intrinsics()
        if args.calibrate:
            teach_calibration(robot, intrinsics)
            return
        calibration = None
        if args.enable_motion:
            if CALIBRATION_FILE.exists():
                calibration = load_calibration(intrinsics)
            else:
                print("Motion disabled: first run this script with --calibrate.")
        else:
            print("READ-ONLY: robot motion disabled; no manual camera teaching needed.")
        print("Workspace is detected automatically from all four Niryo ring markers.")
        print("Board coordinates: M1=(0,0), M2=(1,0), M3=(1,1), M4=(0,1).")
        print("These are image-oriented board coordinates, NOT robot XYZ or millimetres.")
        print("Click a die to print its pixel centre and board position.")
        if args.enable_motion:
            print("M = confirm close-up | B = confirm return to overview")
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

            detections, workspace = detect_dice(frame)
            visible = workspace is not None
            if visible != workspace_visible:
                print(
                    "Workspace found: 4 ring markers."
                    if visible else
                    "Workspace NOT found or ambiguous: show all 4 ring markers. Motion blocked."
                )
                workspace_visible = visible
            if not visible:
                selection["pixel"] = None
            selected = None
            selected_index = None
            if selection["pixel"] is not None:
                for index, die in enumerate(detections, start=1):
                    if cv2.pointPolygonTest(
                        die["contour"], selection["pixel"], False
                    ) >= 0:
                        selected = die
                        selected_index = index
                        break
            if selection["pending"]:
                if selected is None:
                    print("No detected die at the click; all four markers must be visible.")
                else:
                    x_rel, y_rel = selected["workspace_position"]
                    print(
                        f"Die {selected_index}: pixel centre={selected['center']}, "
                        f"board x={x_rel:.4f}, y={y_rel:.4f} (relative, not robot XYZ)."
                    )
                selection["pending"] = False
            output = draw_detections(frame, detections, workspace)
            if selected is not None:
                cv2.drawContours(output, [selected["contour"]], -1, (0, 255, 255), 3)
                cv2.imshow("Selected Die", selected["warped"])
                selected_window_open = True
            elif selected_window_open:
                cv2.destroyWindow("Selected Die")
                selected_window_open = False

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
            elif key in (ord("m"), ord("b")) and not args.enable_motion:
                print("Motion blocked: read-only mode. Only dice positions are enabled.")
            elif key == ord("m"):
                if workspace is None:
                    print("Motion blocked: all four workspace markers must be visible.")
                    continue
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
                    selected_window_open = False
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
