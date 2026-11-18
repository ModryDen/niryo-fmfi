from pathlib import Path
import argparse

from pyniryo import NiryoRobot
import numpy as np
import cv2

from dice_camera_calibration import (
    CALIBRATION_FILE,
    OVERVIEW_ANGLE_TOLERANCE,
    confirmed_move,
    load_calibration,
    read_frame,
    require_live_camera,
    require_pose,
    teach_calibration,
)
from dice_workspace import detect_workspace, find_ring_markers, order_corners
from dice_grip_orientation import make_reference, save_reference, load_reference, die_world_angle, square_angle_difference
from dice_pick_place import (
    PICK_FILE, PickCalibration, load_pick_calibration, teach_pick_calibration,
    repair_pick_calibration, pick_and_place, preview_pick, live_inspection, resolve_pick_target,
    PickupPreflightError, load_taught_target, refresh_fixed_reference, refresh_area_reference,
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



def recover_glare_contours(gray, workspace_mask, contours, threshold):
    """Recover bright die outlines; flag estimates rather than trust a shadow centre."""
    candidates = [(contour, False) for contour in contours]
    edges = cv2.Canny(gray, 40, 100)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    outlines, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    for outline in outlines:
        perimeter = cv2.arcLength(outline, True)
        if perimeter == 0:
            continue
        quad = cv2.approxPolyDP(outline, .04 * perimeter, True)
        if len(quad) == 5:
            quad = cv2.approxPolyDP(outline, .05 * perimeter, True)
        if len(quad) != 4 or not cv2.isContourConvex(quad):
            continue
        area = cv2.contourArea(quad)
        (_, _), (width, height), _ = cv2.minAreaRect(quad)
        if area < MIN_AREA or min(width, height) <= 0 or max(width, height) > MAX_DIE_SIZE:
            continue
        if min(width, height) / max(width, height) < MIN_ASPECT_RATIO:
            continue
        polygon_mask = np.zeros(gray.shape, np.uint8)
        cv2.fillConvexPoly(polygon_mask, quad.reshape(4, 2), 255)
        inside = polygon_mask != 0
        if np.any(inside & (workspace_mask == 0)):
            continue
        values = gray[inside]
        if np.mean(values > 180) < .25 or np.mean(values < threshold) < .20:
            continue
        x, y, w, h = cv2.boundingRect(quad)
        matches = []
        for index, (old, _) in enumerate(candidates):
            if cv2.contourArea(old) < MIN_AREA:
                continue
            ox, oy, ow, oh = cv2.boundingRect(old)
            intersection = max(0, min(x+w, ox+ow)-max(x, ox)) * max(0, min(y+h, oy+oh)-max(y, oy))
            union = w*h + ow*oh - intersection
            if union and intersection / union > .20:
                matches.append(index)
        if len(matches) > 1:
            continue  # Do not combine neighbouring dice into a single outline.
        if matches:
            index = matches[0]
            old = candidates[index][0]
            if area > cv2.contourArea(old) * 1.25:
                candidates[index] = (quad, True)
        else:
            candidates.append((quad, True))
    return candidates


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

    for contour, glare_estimate in recover_glare_contours(gray, workspace_mask, contours, threshold):
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

        if glare_estimate:
            # When glare joins most of the bright top face into one component,
            # use that component instead of a shadow-expanded outer rectangle.
            bright = np.uint8((gray > 180) & (contour_mask != 0))
            count, labels, stats, centers = cv2.connectedComponentsWithStats(bright)
            if count > 1:
                index = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
                bright_area = stats[index, cv2.CC_STAT_AREA]
                total_bright = int(np.count_nonzero(bright))
                if bright_area >= .15 * area and bright_area >= .75 * total_bright:
                    cx, cy = centers[index]
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
            "glare_estimate": glare_estimate,
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
        cv2.polylines(output, [workspace.corners.astype(np.int32)], True, (255, 200, 0), 2)
        for index, midpoint in enumerate(workspace.side_points, start=1):
            point = tuple(np.int32(np.round(midpoint)))
            cv2.drawMarker(output, point, (255, 200, 0), cv2.MARKER_CROSS, 12, 2)
            cv2.putText(output, f"S{index}", (point[0] + 8, point[1] + 16),
                        cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 200, 0), 2)
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
            output, f"Die {index}" + (" GLARE" if die.get("glare_estimate") else ""), (cx + 8, cy - 8),
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



def draw_taught_target(output, target):
    """Show the fixed taught pickup location; never alter detector input pixels."""
    point = tuple(np.rint(target.pixel).astype(int))
    color = (0, 200, 255)
    cv2.drawMarker(output, point, color, cv2.MARKER_CROSS, 26, 2)
    cv2.circle(output, point, 5, color, 1)
    cv2.putText(output, f'TAUGHT {target.number}', (point[0] + 16, point[1] - 20),
                cv2.FONT_HERSHEY_SIMPLEX, .55, color, 2)
    cv2.putText(output, 'Fixed target: place die centre on orange cross (arm stationary)',
                (10, output.shape[0] - 40), cv2.FONT_HERSHEY_SIMPLEX, .43, color, 1)



def draw_pick_area(output, calibration, workspace, selected, image_size):
    """Show the actual calibrated region and selected robot target, without motion."""
    if workspace is None:
        return
    try:
        # Keep the usable-area outline visible when a separate reference check
        # blocks motion. This is a drawing only; target resolution still validates.
        boundary = np.int32(np.round(calibration.boundary_in_workspace(workspace, validate_reference=False)))
        cv2.polylines(output, [boundary], True, (0, 180, 255), 2)
        cv2.putText(output, ('PICK AREA: M1-M4, marker rings excluded' if getattr(calibration, 'board_area', False) else
                             'PICK AREA: any die centre inside orange outline'),
                    (10, output.shape[0] - 40), cv2.FONT_HERSHEY_SIMPLEX, .45, (0, 180, 255), 1)
        calibration.check_board(workspace)
        if selected is not None:
            pose = resolve_pick_target(calibration, selected['center'], image_size, workspace, box=selected['box'])
            text = f'Tool X={pose[0]:.4f} Y={pose[1]:.4f} Z={pose[2]:.4f} m'
            cv2.putText(output, text, (10, output.shape[0] - 60),
                        cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 255, 255), 1)
    except ValueError as error:
        cv2.putText(output, f'Pickup blocked: {str(error)[:85]}',
                    (10, output.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, .4, (0, 0, 255), 1)


def main():
    global reference_gray

    parser = argparse.ArgumentParser(description="Detect dice and teach camera close-up poses.")
    parser.add_argument("--calibrate", action="store_true",
                        help="Teach camera mapping manually; does not move the robot.")
    parser.add_argument("--enable-motion", action="store_true",
                        help="Allow confirmed moves with the relevant taught calibration. "
                             "Default: read-only dice positions, no motion.")
    parser.add_argument("--teach-pick", action="store_true",
                        help="Manually teach gripper pickup and one drop pose; no motion commands.")
    parser.add_argument("--repair-pick", action="store_true",
                        help="Correct individual saved pickup records manually; no motion commands.")
    parser.add_argument("--taught-target", type=int, choices=range(1, 6),
                        help="Use one fixed taught die pose directly, without camera mapping interpolation.")
    parser.add_argument("--refresh-fixed-view", action="store_true",
                        help="Confirm unchanged setup and record marker reference for this fixed-target session only.")
    parser.add_argument("--preview-pick", action="store_true",
                        help="P confirms an upper approach and return only, without descent or gripping.")
    parser.add_argument("--refresh-area-view", action="store_true",
                        help="Confirm a fixed physical setup and register the current camera view for general picking.")
    parser.add_argument("--board-area", action="store_true",
                        help="Extend the planar pickup mapping to M1-M4; verify edge positions with --preview-pick first.")
    parser.add_argument('--preview-clearance-mm', type=float,
                        help='Lower preview to this distance above taught grip Z (minimum 30 mm); pauses until RETURN.')
    parser.add_argument('--auto-yaw', action='store_true', help='Align jaw rotation to cube edges using a reference taught with Y.')
    parser.add_argument('--preview-speed', type=int, default=None, choices=range(1,51),
                        help='Upper approach speed for preview, default 50 percent; inspection, rise and return use 20 percent.')
    parser.add_argument('--live-align', action='store_true', help='Live manual XY alignment at lower preview height; no descent or grip.')
    args = parser.parse_args()
    if args.preview_speed is None:
        args.preview_speed = 50 if args.preview_pick else 10
    if args.live_align and (not args.preview_pick or args.preview_clearance_mm is None):
        parser.error('--live-align requires --preview-pick and --preview-clearance-mm.')
    if args.preview_speed != 10 and not args.preview_pick:
        parser.error('--preview-speed is available only with --preview-pick.')
    if args.preview_clearance_mm is not None and (not args.preview_pick or not np.isfinite(args.preview_clearance_mm) or args.preview_clearance_mm < 30):
        parser.error('--preview-clearance-mm requires --preview-pick and a value of at least 30.')
    if args.auto_yaw and (not args.enable_motion or not args.preview_pick or args.taught_target is not None):
        parser.error('--auto-yaw currently requires general --preview-pick mode; verify rotation and tool offset before gripping.')
    if args.board_area and (not args.enable_motion or args.taught_target is not None):
        parser.error("--board-area requires --enable-motion and general mode.")
    if args.refresh_area_view and (not args.enable_motion or args.taught_target is not None):
        parser.error("--refresh-area-view requires --enable-motion and general mode (no --taught-target).")
    if args.preview_pick and not args.enable_motion:
        parser.error("--preview-pick requires --enable-motion because the arm will move.")
    if args.refresh_fixed_view and args.taught_target is None:
        parser.error("--refresh-fixed-view requires --taught-target.")
    if args.taught_target is not None and not args.enable_motion:
        parser.error("--taught-target requires --enable-motion.")
    if sum((args.calibrate, args.teach_pick, args.repair_pick)) > 1:
        parser.error("Choose only one teaching/repair mode.")
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
        if args.repair_pick:
            repair_pick_calibration(robot, intrinsics)
            return
        if args.teach_pick:
            teach_pick_calibration(robot, intrinsics)
            return
        if args.calibrate:
            teach_calibration(robot, intrinsics)
            return
        pick_calibration = None
        if args.enable_motion and args.taught_target is not None:
            pick_calibration = load_taught_target(intrinsics, args.taught_target)
            if args.refresh_fixed_view:
                if not refresh_fixed_reference(robot, pick_calibration, intrinsics, ROBOT_IP):
                    return
            print(f"FIXED TAUGHT TARGET {args.taught_target}: uses its recorded tool pose directly.")
            print("Keep that die in its taught position. This mode does not map arbitrary dice positions. Frame Die numbers can change; TAUGHT marks the fixed location.")
        elif args.enable_motion and PICK_FILE.exists():
            try:
                pick_calibration = load_pick_calibration(intrinsics)
                if args.auto_yaw:
                    pick_calibration.auto_yaw = True
                    try:
                        pick_calibration.yaw_reference = load_reference(pick_calibration)
                    except (ValueError, OSError, KeyError) as error:
                        pick_calibration.yaw_reference = None
                        print(f'Yaw reference unavailable: {error}')
                    print('Y = teach jaw reference: reference die edge must be parallel to actual jaw closing axis.')
                if args.board_area:
                    pick_calibration.enable_board_area()
                    print("BOARD AREA: M1-M4. Outside the original taught hull this is planar extrapolation; "
                          "verify upper approaches at the edges before gripping.")
                if args.refresh_area_view:
                    if not refresh_area_reference(robot, pick_calibration, intrinsics, ROBOT_IP):
                        return
                print(f"AREA PICK: any detected die inside the selected orange boundary. "
                      f"Independent fifth-point error: {pick_calibration.error * 1000:.2f} mm.")
            except (ValueError, KeyError, OSError) as error:
                pick_calibration = None
                print(f"Pickup disabled: {error}")
                if args.refresh_area_view:
                    return
        calibration = None
        if args.enable_motion:
            if CALIBRATION_FILE.exists():
                try:
                    calibration = load_calibration(intrinsics)
                except (ValueError, KeyError, OSError) as error:
                    print(f"Camera close-ups disabled: {error}")
            else:
                print("Camera close-ups disabled: teach with --calibrate if needed.")
        else:
            print("READ-ONLY: robot motion disabled; no manual camera teaching needed.")
        print("Workspace is detected automatically from all four Niryo ring markers.")
        print("Board coordinates: M1=(0,0), M2=(1,0), M3=(1,1), M4=(0,1).")
        print("These are image-oriented board coordinates, NOT robot XYZ or millimetres.")
        print("Click a die to print its pixel centre and board position.")
        if args.enable_motion:
            print("M = confirm close-up | B = confirm return | P = " +
                  ("preview upper approach (no grip)" if args.preview_pick else "pick selected die to taught drop"))
            if pick_calibration is None:
                print("Pickup needs separate teaching: run with --teach-pick first.")
        print("V = refresh area view (no motion) | Q = quit | R = reset alignment | S = save")
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
            if args.taught_target is not None and pick_calibration is not None:
                draw_taught_target(output, pick_calibration)
            if isinstance(pick_calibration, PickCalibration):
                draw_pick_area(output, pick_calibration, workspace, selected,
                               (frame.shape[1], frame.shape[0]))
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
            elif key in (ord("m"), ord("b"), ord("p")) and not args.enable_motion:
                print("Motion blocked: read-only mode. Only dice positions are enabled.")
            elif key == ord('v'):
                if not isinstance(pick_calibration, PickCalibration):
                    print('V needs a loaded general pickup calibration.')
                    continue
                try:
                    if refresh_area_reference(robot, pick_calibration, intrinsics, ROBOT_IP):
                        selection['pixel'] = None
                        print('View refreshed. Select the next die.')
                except ValueError as error:
                    print(f'View refresh blocked: {error}')
            elif key == ord('y') and args.auto_yaw:
                if selected is None or workspace is None or not isinstance(pick_calibration, PickCalibration):
                    print('Select a reference die first.')
                    continue
                if selected.get('glare_estimate'):
                    print('Yaw reference blocked: resolve glare first.')
                    continue
                try:
                    require_live_camera(ROBOT_IP)
                    actual = robot.get_pose().to_list()
                    require_pose(actual, pick_calibration.overview, angle_tolerance=OVERVIEW_ANGLE_TOLERANCE)
                    print('Place a reference cube with one edge PARALLEL to the actual open jaw closing axis.')
                    print('Re-select the cube after positioning it. This teaches alignment, not a new robot pose.')
                    if input('Type ALIGNED only if the selected cube is already aligned; anything else cancels: ').strip() != 'ALIGNED':
                        continue
                    fresh = read_frame(robot, intrinsics)
                    fresh_dice, fresh_workspace = detect_dice(fresh)
                    pick_calibration.check_board(fresh_workspace)
                    nearest = min(fresh_dice, key=lambda d: np.linalg.norm(np.asarray(d['center'])-selected['center']), default=None)
                    if nearest is None or nearest.get('glare_estimate') or np.linalg.norm(np.asarray(nearest['center'])-selected['center']) > 3:
                        raise ValueError('Reference die moved or became uncertain; re-select it.')
                    require_pose(robot.get_pose().to_list(), actual)
                    reference = make_reference(pick_calibration, nearest['box'], fresh_workspace, actual)
                    save_reference(reference)
                    pick_calibration.yaw_reference = reference
                    print('Jaw reference saved. Auto rotation uses the closest equivalent square orientation.')
                except ValueError as error:
                    print(f'Yaw reference blocked: {error}')
            elif key == ord("p"):
                if pick_calibration is None or workspace is None or selected is None:
                    print("Pickup blocked: need --teach-pick calibration, four markers and a selected die.")
                    continue
                if selected.get('glare_estimate') and (not args.preview_pick or args.auto_yaw):
                    print('Pickup blocked: glare makes this die centre uncertain. Improve lighting and verify in PREVIEW.')
                    continue
                try:
                    require_live_camera(ROBOT_IP)
                    pick_calibration.check_board(workspace)
                    # Validate a fresh frame AFTER terminal confirmation, before any actuator command.
                    def revalidate():
                        require_live_camera(ROBOT_IP)
                        fresh = read_frame(robot, intrinsics)
                        if fresh.shape != frame.shape:
                            raise ValueError("Camera resolution changed.")
                        fresh_dice, fresh_workspace = detect_dice(fresh)
                        pick_calibration.check_board(fresh_workspace)
                        if len(fresh_dice) != len(detections):
                            raise ValueError("Dice changed during confirmation. Select again.")
                        for old, new in zip(detections, fresh_dice):
                            if old is selected and args.auto_yaw:
                                before = die_world_angle(pick_calibration, old['box'], workspace)
                                after = die_world_angle(pick_calibration, new['box'], fresh_workspace)
                                if new.get('glare_estimate') or abs(square_angle_difference(before, after)) > np.deg2rad(5):
                                    raise ValueError('Selected die rotation changed or became uncertain.')
                            if old is selected and new.get('glare_estimate') and not args.preview_pick:
                                raise ValueError('Selected die became glare-uncertain during confirmation.')
                            if (np.linalg.norm(np.asarray(old["center"]) - new["center"]) > 3
                                    or abs(new["area"] / old["area"] - 1) > .15):
                                raise ValueError("Dice moved during confirmation. Select again.")
                    # Errors after the first actuator command propagate and end the session.
                    target = resolve_pick_target(pick_calibration, selected["center"],
                                                 (frame.shape[1], frame.shape[0]), workspace, box=selected["box"])
                    print('Selected die -> calculated tool XYZ/RPY:', np.round(target, 6).tolist())
                    require_pose(robot.get_pose().to_list(), pick_calibration.overview,
                                 angle_tolerance=OVERVIEW_ANGLE_TOLERANCE)
                except ValueError as error:
                    print(f"Pickup blocked: {error}")
                    continue
                try:
                    execute = preview_pick if args.preview_pick else pick_and_place
                    options = dict(workspace=workspace, box=selected['box'])
                    if args.preview_pick:
                        options['clearance_mm'] = args.preview_clearance_mm
                        options['speed_percent'] = args.preview_speed
                        if args.live_align:
                            options['inspection_callback'] = lambda pose: live_inspection(robot, pose, intrinsics, ROBOT_IP)
                    completed = execute(robot, pick_calibration, selected["center"],
                                        (frame.shape[1], frame.shape[0]), revalidate, **options)
                except PickupPreflightError as error:
                    selection["pixel"] = None
                    print(f"Pickup cancelled before motion: {error}")
                    print("Camera remains active. Wait for a stable view, then select the die again.")
                    continue
                if completed:
                    selection["pixel"] = None
                    print("Preview complete: no gripping was performed." if args.preview_pick else
                          "Pick/place commands complete; verify the die was transferred.")
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
