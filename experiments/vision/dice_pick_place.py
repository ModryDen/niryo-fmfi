from pathlib import Path
import json

import cv2
import numpy as np

from dice_camera_calibration import CameraCalibration, pose_array, read_frame, require_pose, same_orientation, orientation_distance, ANGLE_TOLERANCE, require_live_camera, move_linear_pose_compatible, wait_for_pose, OVERVIEW_ANGLE_TOLERANCE
from dice_workspace import detect_workspace, find_ring_markers

PICK_FILE = Path(__file__).with_name('dice_pick_calibration.json')
PICK_DRAFT_FILE = Path(__file__).with_name('dice_pick_draft.json')
# Gentle first trials with light dice; avoid SDK defaults of full peak torque
# and 20-30% continuous holding effort at the mechanical stops.
GRIPPER_OPTIONS = dict(speed=300, max_torque_percentage=30, hold_torque_percentage=10)


def save_pick_draft(data):
    # Drafts never enable motion; the validated calibration is a separate file.
    temporary = PICK_DRAFT_FILE.with_suffix('.json.tmp')
    with temporary.open('w', encoding='utf-8') as stream:
        json.dump(data, stream, indent=2)
    temporary.replace(PICK_DRAFT_FILE)




def require_board_reference(workspace, reference, frame=None):
    if workspace is None:
        detail = ''
        if frame is not None:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            count = len(find_ring_markers(gray))
            detail = f' Found {count} ring candidates; need exactly 4 with valid geometry.'
        raise ValueError('Board detection failed.' + detail + ' No saved records changed.')
    reference = np.asarray(reference, dtype=float)
    distances = np.linalg.norm(workspace.corners - reference, axis=1)
    if np.max(distances) > 3:
        raise ValueError(
            'Board detected: all 4 markers, but image positions differ from the reference. '
            f'M1-M4 differences (pixels): {np.round(distances, 2).tolist()}; limit 3. '
            f'Current corners: {workspace.corners.tolist()}; reference: {reference.tolist()}. '
            'This can reflect camera pose drift or board movement; do not move the board to match pixels.'
        )


def gripper_name(robot):
    name = robot.get_current_tool_id().name
    if not name.startswith('GRIPPER_'):
        raise ValueError('Attach and configure a supported gripper in Niryo Studio first.')
    return name


class PickCalibration(CameraCalibration):
    """Image centres mapped to taught gripper poses, never camera close-up poses."""

    def __init__(self, data):
        if data.get('purpose') != 'dice_pick_place':
            raise ValueError('Expected a separate gripper pick/place calibration.')
        overview = pose_array(data.get('pickup_reference_pose', data['overview']))
        for index, pose in enumerate(data['poses'], 1):
            angle = orientation_distance(overview, pose)
            if angle > ANGLE_TOLERANCE:
                raise ValueError(
                    f'Target {index}: tool orientation differs from overview by '
                    f'{np.degrees(angle):.2f} degrees (limit {np.degrees(ANGLE_TOLERANCE):.2f}). '
                    'Recorded poses remain in dice_pick_draft.json after teaching.'
                )
        mapping_data = dict(data)
        mapping_data['overview'] = overview.tolist()
        super().__init__(mapping_data)
        self.pickup_reference = overview
        self.overview = pose_array(data['overview'])
        if self.overview[2] <= max(pose[2] for pose in self.poses) + .002:
            raise ValueError('Camera overview must remain above pickup height.')
        self.drop = pose_array(data['drop'])
        self.board = np.asarray(data['board_corners'], dtype=np.float32)
        self.view_board = self.board.copy()
        self.tool = data['tool']
        if self.board.shape != (4, 2) or not np.isfinite(self.board).all():
            raise ValueError('Invalid reference board corners.')
        if not same_orientation(self.drop, self.pickup_reference):
            raise ValueError('Keep the same tool orientation at pickup, transfer and drop.')
        if self.drop[2] >= self.overview[2] - 0.002:
            raise ValueError('Drop pose must be below the taught transfer height.')

    def enable_board_area(self):
        """Explicitly extend the validated planar model to the M1-M4 polygon."""
        corners = self.board.astype(np.float32)
        if not cv2.isContourConvex(corners.reshape(-1, 1, 2)) or cv2.contourArea(corners) < 2500:
            raise ValueError('Invalid M1-M4 board polygon.')
        denominators = np.column_stack((corners, np.ones(4))) @ self.matrix[2]
        if not (np.all(denominators > 1e-8) or np.all(denominators < -1e-8)):
            raise ValueError('Mapping has a projection singularity inside the extended board.')
        projected = cv2.perspectiveTransform(corners.reshape(1, 4, 2), self.matrix)[0]
        if not np.isfinite(projected).all() or not cv2.isContourConvex(projected.reshape(-1, 1, 2)):
            raise ValueError('Invalid robot-coordinate projection of the board.')
        self.boundary = corners.copy()
        self.board_area = True

    def target(self, pixel, image_size):
        target = super().target(pixel, image_size)
        # Camera overview orientation and physically taught grip orientation differ.
        target[3:] = self.pickup_reference[3:]
        return target

    def target_in_workspace(self, pixel, image_size, workspace):
        self.check_board(workspace)
        point = np.asarray(pixel, dtype=np.float32)
        if point.shape != (2,) or not np.isfinite(point).all():
            raise ValueError('Target pixels must be finite XY coordinates.')
        # Use marker correspondences to express the current pixel in the original
        # calibration image. This does not allow a physically moved board.
        matrix = cv2.getPerspectiveTransform(workspace.corners.astype(np.float32), self.board)
        original_pixel = cv2.perspectiveTransform(point.reshape(1, 1, 2), matrix)[0, 0]
        try:
            return self.target(original_pixel, image_size)
        except ValueError as error:
            if 'outside the calibrated area' in str(error):
                raise ValueError('Die centre is outside the selected pickup area (orange outline). '
                                 'Use --board-area for the full M1-M4 board; marker rings remain excluded.') from error
            raise

    def boundary_in_workspace(self, workspace, validate_reference=True):
        if validate_reference:
            self.check_board(workspace)
        elif workspace is None:
            raise ValueError('Workspace is not visible.')
        matrix = cv2.getPerspectiveTransform(self.board, workspace.corners.astype(np.float32))
        return cv2.perspectiveTransform(self.boundary.reshape(1, -1, 2), matrix)[0]

    def check_board(self, workspace):
        require_board_reference(workspace, self.view_board)


def resolve_pick_target(calibration, pixel, image_size, workspace=None, box=None):
    if isinstance(calibration, PickCalibration) and workspace is not None:
        target = calibration.target_in_workspace(pixel, image_size, workspace)
    else:
        target = calibration.target(pixel, image_size)
    if getattr(calibration, 'auto_yaw', False):
        from dice_grip_orientation import apply_die_yaw
        if box is None or workspace is None:
            raise ValueError('Auto rotation requires a current die outline and board.')
        target = apply_die_yaw(calibration, target, box, workspace)
    return target


def load_pick_calibration(intrinsics):
    with PICK_FILE.open(encoding='utf-8') as stream:
        result = PickCalibration(json.load(stream))
    for actual, saved in zip(intrinsics, result.intrinsics):
        if np.shape(actual) != saved.shape or not np.allclose(actual, saved):
            raise ValueError('Camera intrinsics changed. Re-teach pickup.')
    return result



class TaughtTargetPickup:
    """One fixed, physically taught target. No image-to-robot interpolation."""

    def __init__(self, data, number):
        if data.get('purpose') != 'dice_pick_place' or number not in range(1, 6):
            raise ValueError('Select a recorded pickup target from 1 to 5.')
        self.overview = pose_array(data['overview'])
        self.pickup = pose_array(data['poses'][number - 1])
        self.drop = pose_array(data['drop'])
        self.tool = data['tool']
        self.number = number
        self.image_size = tuple(data['image_size'])
        self.pixel = np.asarray(data['pixels'][number - 1], dtype=float)
        self.board = np.asarray(data['board_corners'], dtype=float)
        self.intrinsics = (np.asarray(data['camera_matrix']), np.asarray(data['distortion']))
        if self.pixel.shape != (2,) or not np.isfinite(self.pixel).all():
            raise ValueError('Invalid recorded target pixel.')
        if self.board.shape != (4, 2) or not np.isfinite(self.board).all():
            raise ValueError('Invalid board reference.')
        if self.overview[2] <= max(self.pickup[2], self.drop[2]) + .002:
            raise ValueError('Overview must clear the pickup and drop heights.')
        reference = pose_array(data.get('pickup_reference_pose', data['overview']))
        if not same_orientation(reference, self.pickup) or not same_orientation(reference, self.drop):
            raise ValueError('Recorded pickup/drop orientation is inconsistent.')

    def check_board(self, workspace):
        require_board_reference(workspace, self.board)

    def target(self, pixel, image_size):
        if tuple(image_size) != self.image_size:
            raise ValueError('Camera image size changed.')
        point = np.asarray(pixel, dtype=float)
        if point.shape != (2,) or not np.isfinite(point).all():
            raise ValueError('Invalid selected die centre.')
        distance = float(np.linalg.norm(point - self.pixel))
        if distance > 5:
            raise ValueError(
                f'Fixed target {self.number}: centre is {distance:.1f} px from the taught position '
                f'(limit 5). Selected {point.tolist()}, expected {self.pixel.tolist()}. '
                'With arm stationary, place the die centre on the orange TAUGHT cross. '
                'Frame Die numbers are not taught target numbers.'
            )
        return self.pickup.copy()


def load_taught_target(intrinsics, number):
    with PICK_DRAFT_FILE.open(encoding='utf-8') as stream:
        result = TaughtTargetPickup(json.load(stream), number)
    for actual, saved in zip(intrinsics, result.intrinsics):
        if np.shape(actual) != saved.shape or not np.allclose(actual, saved):
            raise ValueError('Camera intrinsics changed.')
    return result



def refresh_fixed_reference(robot, target, intrinsics, ip_address, confirm=input):
    """Explicit session-only image reference for one unchanged taught target."""
    if not isinstance(target, TaughtTargetPickup):
        raise ValueError('Session reference refresh is only for fixed taught targets.')
    require_gripper_healthy(robot)
    if robot.get_learning_mode():
        raise ValueError('Disable Learning Mode first.')
    require_pose(robot.get_pose().to_list(), target.overview,
                 angle_tolerance=OVERVIEW_ANGLE_TOLERANCE)
    print('Fixed-view refresh changes ONLY this session marker reference, not robot pickup/drop poses.')
    print('Use only if board, robot base, camera mounting and taught die position are unchanged.')
    if confirm('Type FIXED SETUP to confirm, anything else cancels: ').strip() != 'FIXED SETUP':
        return False
    require_live_camera(ip_address)
    actual = pose_array(robot.get_pose().to_list())
    require_pose(actual, target.overview, angle_tolerance=OVERVIEW_ANGLE_TOLERANCE)
    frame = read_frame(robot, intrinsics)
    if (frame.shape[1], frame.shape[0]) != target.image_size:
        raise ValueError('Camera image size changed.')
    workspace = detect_workspace(frame)
    if workspace is None:
        require_board_reference(None, target.board, frame)
    require_gripper_healthy(robot)
    require_pose(robot.get_pose().to_list(), actual, angle_tolerance=OVERVIEW_ANGLE_TOLERANCE)
    target.board = workspace.corners.copy()
    print('Current session marker reference recorded. Target centre must still match within 5 pixels.')
    print('No motion performed. No calibration files changed.')
    return True



def refresh_area_reference(robot, calibration, intrinsics, ip_address, confirm=input):
    """Session-only camera reference; retain the canonical image-to-tool mapping."""
    if not isinstance(calibration, PickCalibration):
        raise ValueError('Area refresh requires a validated general pickup calibration.')
    require_gripper_healthy(robot)
    if robot.get_learning_mode():
        raise ValueError('Disable Learning Mode first.')
    # This is explicit image-reference teaching, not a movement endpoint check.
    # A modest view rotation is registered through the four markers; XYZ still
    # must match within 2 mm. Bound rotation to preserve marker correspondence.
    capture_angle_limit = np.deg2rad(10)
    require_pose(robot.get_pose().to_list(), calibration.overview,
                 angle_tolerance=capture_angle_limit)
    print('AREA VIEW: update the camera view for this session; robot pickup/drop teaching is unchanged.')
    print('Confirm that board, robot base, camera/tool mounting and die height match the validated teaching.')
    print('Dice may be at new positions inside the area. If the board or mounting moved, cancel and re-teach.')
    if confirm('Type BOARD FIXED to confirm, anything else cancels: ').strip() != 'BOARD FIXED':
        return False
    require_live_camera(ip_address)
    actual = pose_array(robot.get_pose().to_list())
    require_pose(actual, calibration.overview, angle_tolerance=capture_angle_limit)
    frame = read_frame(robot, intrinsics)
    if (frame.shape[1], frame.shape[0]) != calibration.image_size:
        raise ValueError('Camera image size changed.')
    workspace = detect_workspace(frame)
    if workspace is None:
        require_board_reference(None, calibration.board, frame)
    require_gripper_healthy(robot)
    require_pose(robot.get_pose().to_list(), actual, angle_tolerance=OVERVIEW_ANGLE_TOLERANCE)
    # Never replace calibration.board: it anchors the validated homography.
    # Live pixels are registered to that original board before robot XY conversion.
    calibration.view_board = workspace.corners.copy()
    calibration.overview = actual.copy()
    print('Area view recorded for this session. Choose any detected die inside the orange boundary.')
    print('No motion performed and no calibration files changed.')
    return True


def teach_pick_calibration(robot, intrinsics):
    print('PICK TEACHING: no automatic arm or gripper commands.')
    print('Use five stationary dice of the working height: four outer targets and one inside.')
    print('Start at an overview pose whose Z clears the whole transfer route, including the drop area.')
    print('Keep board, dice and camera/tool mounting fixed. Use the SAME tool orientation throughout.')
    print('In Studio, manually centre the open gripper on each die at pickup height; do not centre the camera.')
    if input('Type TEACH PICK to begin: ').strip() != 'TEACH PICK':
        return
    tool = gripper_name(robot)
    overview = robot.get_pose().to_list()
    frame = read_frame(robot, intrinsics)
    workspace = detect_workspace(frame)
    if workspace is None:
        raise ValueError('All four board markers must be visible at overview.')
    points = []
    window = 'Pick teaching: 4 outer die centres, then 1 inside; Enter accepts, R resets, Q cancels'

    def click(event, x, y, _flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 5:
            if workspace.mask[y, x]:
                points.append((x, y))

    cv2.namedWindow(window)
    cv2.setMouseCallback(window, click)
    try:
        while True:
            image = frame.copy()
            for index, point in enumerate(points, 1):
                cv2.circle(image, point, 5, (0, 0, 255), -1)
                cv2.putText(image, str(index), point, cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 255, 0), 2)
            cv2.imshow(window, image)
            key = cv2.waitKey(20) & 255
            if key == ord('q'):
                return
            if key == ord('r'):
                points.clear()
            if key in (10, 13) and len(points) == 5:
                break
    finally:
        cv2.destroyWindow(window)
    poses = []
    data = dict(version=1, purpose='dice_pick_place', image_size=[frame.shape[1], frame.shape[0]],
                overview=overview, pixels=points, poses=poses,
                camera_matrix=np.asarray(intrinsics[0]).tolist(),
                distortion=np.asarray(intrinsics[1]).tolist(),
                board_corners=workspace.corners.tolist(), tool=tool)
    save_pick_draft(data)
    print(f'Each recorded pose is saved to draft: {PICK_DRAFT_FILE}')
    for index, pixel in enumerate(points, 1):
        print(f'Target {index}, overview pixel {pixel}: manually centre gripper at pickup height.')
        if input('Release manual control; type RECORD (anything else cancels): ').strip() != 'RECORD':
            return
        poses.append(robot.get_pose().to_list())
        save_pick_draft(data)
        angle = np.degrees(orientation_distance(overview, poses[-1]))
        print(f'Saved target {index}; tool orientation difference: {angle:.2f} degrees.')
    print('Manually teach one drop pose with clearance to open the gripper. Same orientation.')
    if input('Release manual control; type RECORD DROP: ').strip() != 'RECORD DROP':
        return
    drop = robot.get_pose().to_list()
    data['drop'] = drop
    save_pick_draft(data)
    result = PickCalibration(data)
    if gripper_name(robot) != tool:
        raise ValueError('Tool changed during teaching.')
    print('Return manually to the original overview; keep the five dice fixed for validation.')
    if input('Type CHECK when ready: ').strip() != 'CHECK':
        return
    require_pose(robot.get_pose().to_list(), overview)
    result.check_board(detect_workspace(read_frame(robot, intrinsics)))
    with PICK_FILE.open('w', encoding='utf-8') as stream:
        json.dump(data, stream, indent=2)
    print(f'Pickup mapping saved: {PICK_FILE}; fifth-point error {result.error * 1000:.1f} mm.')



def replace_pick_record(data, label, pose):
    """Replace one physically re-taught pose; never manufacture orientation values."""
    from copy import deepcopy
    pose = pose_array(pose)
    angle = orientation_distance(data.get('pickup_reference_pose', data['overview']), pose)
    if angle > ANGLE_TOLERANCE:
        raise ValueError(
            f'Not recorded: actual orientation differs by {np.degrees(angle):.2f} degrees '
            f'(limit {np.degrees(ANGLE_TOLERANCE):.2f}). '
            'Use Pose fields in Studio, then check the actual orientation.'
        )
    result = deepcopy(data)
    if label == 'D':
        if pose[2] >= pose_array(data['overview'])[2] - .002:
            raise ValueError('Drop must be below transfer height.')
        result['drop'] = pose.tolist()
    else:
        index = int(label) - 1
        if index not in range(5) or len(data['poses']) != 5:
            raise ValueError('Need a complete five-point draft and target number 1-5.')
        heights = [p[2] for i, p in enumerate(data['poses']) if i != index] + [pose[2]]
        if np.ptp(heights) > .002:
            raise ValueError('Not recorded: pickup Z must match the other targets within 2 mm.')
        result['poses'][index] = pose.tolist()
    return result



def replace_pick_pixels(data, points, workspace=None, refresh_reference=False):
    """Update observed image centres, preserving every measured tool pose."""
    from copy import deepcopy
    points = np.asarray(points, dtype=np.float32)
    if points.shape != (5, 2) or not np.isfinite(points).all():
        raise ValueError('Select exactly five finite image centres.')
    if np.any(points < 0) or np.any(points >= data['image_size']):
        raise ValueError('Image points must be inside the frame.')
    boundary = cv2.convexHull(points[:4]).reshape(-1, 2)
    if len(boundary) != 4 or cv2.contourArea(boundary) < 2500:
        raise ValueError('First four centres must surround a convex working area.')
    if cv2.pointPolygonTest(boundary, tuple(map(float, points[4])), False) <= 0:
        raise ValueError('The fifth centre must be inside the four outer centres.')
    result = deepcopy(data)
    result['pixels'] = points.tolist()
    if workspace is not None:
        # Image points and marker reference must describe the SAME frozen frame.
        if not refresh_reference:
            require_board_reference(workspace, data['board_corners'])
        for point in points:
            x, y = np.rint(point).astype(int)
            if not (0 <= y < workspace.mask.shape[0] and 0 <= x < workspace.mask.shape[1]
                    and workspace.mask[y, x]):
                raise ValueError('All die centres must be inside the detected usable board.')
        result['board_corners'] = workspace.corners.tolist()
    return result


def reselect_pick_pixels(robot, intrinsics, data, refresh_reference=False):
    capture_pose = pose_array(robot.get_pose().to_list())
    if refresh_reference:
        # REFRESH defines a NEW image reference at the actual nearby overview.
        # It must not substitute this orientation into taught pickup poses.
        distance = np.linalg.norm(capture_pose[:3] - pose_array(data['overview'])[:3])
        if distance > .002:
            raise ValueError(f'New overview position differs by {distance * 1000:.2f} mm; limit 2 mm.')
    else:
        require_pose(capture_pose, data['overview'])
    frame = read_frame(robot, intrinsics)
    if (frame.shape[1], frame.shape[0]) != tuple(data['image_size']):
        raise ValueError('Camera image size changed.')
    workspace = detect_workspace(frame)
    reference = np.asarray(data['board_corners'])
    if refresh_reference:
        # Re-observe the image mapping only after explicit confirmation of a fixed
        # physical setup. Never infer that the board is fixed from image drift.
        if workspace is None:
            require_board_reference(workspace, reference, frame)
        print('REFRESH records the actual camera overview, image centres and markers. Pickup/drop poses stay unchanged.')
        print('Use ONLY if the board, all five dice, robot base, camera and tool mounting are unchanged.')
        print('If anything moved physically, cancel; the saved robot poses may no longer apply.')
        if input('Type SETUP UNCHANGED to confirm, anything else cancels: ').strip() != 'SETUP UNCHANGED':
            return data
    else:
        require_board_reference(workspace, reference, frame)
    print('IMAGE: keep all five dice at their original physical positions.')
    print('Click actual die centres in original order 1,2,3,4,5. Old labels are hints only.')
    print('R resets new clicks; Enter accepts five; Q cancels without changes.')
    points = []
    window = 'Repair image centres: same dice 1-5; Enter accepts, R resets, Q cancels'

    def click(event, x, y, _flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 5 and workspace.mask[y, x]:
            points.append((x, y))

    cv2.namedWindow(window)
    cv2.setMouseCallback(window, click)
    try:
        while True:
            image = frame.copy()
            for index, pixel in enumerate(data['pixels'], 1):
                point = tuple(map(int, pixel))
                cv2.drawMarker(image, point, (0, 180, 255), cv2.MARKER_CROSS, 12, 1)
                cv2.putText(image, f'old {index}', (point[0] + 8, point[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, .4, (0, 180, 255), 1)
            for index, pixel in enumerate(points, 1):
                cv2.circle(image, pixel, 4, (0, 255, 0), 1)
                cv2.putText(image, str(index), (pixel[0] + 8, pixel[1] + 16),
                            cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 255, 0), 1)
            cv2.imshow(window, image)
            key = cv2.waitKey(20) & 255
            if key == ord('q'):
                return data
            if key == ord('r'):
                points.clear()
            if key in (10, 13) and len(points) == 5:
                updated = replace_pick_pixels(data, points, workspace, refresh_reference=refresh_reference)
                require_pose(robot.get_pose().to_list(), capture_pose)
                if refresh_reference:
                    updated['pickup_reference_pose'] = data.get('pickup_reference_pose', data['overview'])
                    updated['overview'] = capture_pose.tolist()
                save_pick_draft(updated)
                print('New image centres AND matching board reference saved. Tool poses unchanged. Run CHECK for full validation.')
                return updated
    finally:
        cv2.destroyWindow(window)



def view_pick_draft(robot, intrinsics, data):
    """Diagnostic camera view at any pose; no edits, saving or motion."""
    actual = robot.get_pose().to_list()
    print('VIEW ONLY: no calibration changes and no motion commands.')
    try:
        require_pose(actual, data['overview'])
        print('Current pose matches the original overview.')
    except ValueError as error:
        print(error)
    frame = read_frame(robot, intrinsics)
    workspace = detect_workspace(frame)
    if 'board_corners' in data:
        try:
            require_board_reference(workspace, data['board_corners'], frame)
            print('All 4 board markers match the reference within 3 pixels.')
        except ValueError as error:
            print(error)
    window = 'VIEW ONLY - old image points; Q closes'
    image = frame.copy()
    if workspace is not None:
        for index, corner in enumerate(workspace.corners, 1):
            point = tuple(map(int, corner))
            cv2.circle(image, point, 5, (0, 255, 0), 1)
            cv2.putText(image, f'M{index}', (point[0] + 8, point[1] + 14),
                        cv2.FONT_HERSHEY_SIMPLEX, .4, (0, 255, 0), 1)
    cv2.putText(image, 'VIEW ONLY: old points may shift with camera pose', (8, 20),
                cv2.FONT_HERSHEY_SIMPLEX, .45, (0, 0, 255), 1)
    for index, pixel in enumerate(data['pixels'], 1):
        point = tuple(map(int, pixel))
        cv2.drawMarker(image, point, (0, 180, 255), cv2.MARKER_CROSS, 12, 1)
        cv2.putText(image, f'old {index}', (point[0] + 8, point[1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, .45, (0, 180, 255), 1)
    cv2.namedWindow(window)
    try:
        while True:
            cv2.imshow(window, image)
            if cv2.waitKey(20) & 255 in (ord('q'), 27):
                return
    finally:
        cv2.destroyWindow(window)


def repair_pick_calibration(robot, intrinsics):
    """Manual correction of a saved draft. No arm/gripper commands."""
    from datetime import datetime
    with PICK_DRAFT_FILE.open(encoding='utf-8') as stream:
        data = json.load(stream)
    if data.get('purpose') != 'dice_pick_place' or len(data.get('poses', [])) != 5:
        raise ValueError('Need a saved complete pickup draft.')
    if gripper_name(robot) != data['tool']:
        raise ValueError('Configured gripper differs from the draft.')
    for actual, saved in zip(intrinsics, (data['camera_matrix'], data['distortion'])):
        if np.shape(actual) != np.shape(saved) or not np.allclose(actual, saved):
            raise ValueError('Camera intrinsics changed since teaching.')
    backup = PICK_DRAFT_FILE.with_name(
        'dice_pick_draft_backup_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f') + '.json')
    with backup.open('x', encoding='utf-8') as stream:
        json.dump(data, stream, indent=2)
    print('MANUAL REPAIR ONLY: this mode sends NO arm or gripper commands.')
    print(f'Original records backed up: {backup}')
    print('Keep the board and all five dice fixed. IMAGE can correct image centres at original overview.')
    print('Required tool Roll/Pitch/Yaw:', np.round(data.get('pickup_reference_pose', data['overview'])[3:], 6).tolist())
    while True:
        print('Saved pickup targets (pixel; tool XYZ; orientation difference):')
        for index, (pixel, pose) in enumerate(zip(data['pixels'], data['poses']), 1):
            angle = np.degrees(orientation_distance(data.get('pickup_reference_pose', data['overview']), pose))
            print(f'{index}: {pixel}; {np.round(pose[:3], 6).tolist()}; {angle:.2f} deg')
        command = input('Choose 1-5 for target, D for drop, IMAGE for image centres, REFRESH for fixed setup, VIEW for camera only, CHECK to validate, Q to exit: ').strip().upper()
        if command == 'Q':
            return
        if command == 'VIEW':
            view_pick_draft(robot, intrinsics, data)
            continue
        if command in ('IMAGE', 'REFRESH'):
            try:
                data = reselect_pick_pixels(robot, intrinsics, data,
                                            refresh_reference=(command == 'REFRESH'))
            except ValueError as error:
                print(f'Image correction blocked: {error}')
            continue
        if command == 'CHECK':
            try:
                result = PickCalibration(data)
                require_pose(robot.get_pose().to_list(), result.overview)
                frame = read_frame(robot, intrinsics)
                if (frame.shape[1], frame.shape[0]) != result.image_size:
                    raise ValueError('Camera image size changed.')
                result.check_board(detect_workspace(frame))
                if gripper_name(robot) != result.tool:
                    raise ValueError('Gripper changed.')
            except (ValueError, KeyError) as error:
                print(f'Not ready: {error}')
                print('Corrections remain saved. CHECK requires the latest recorded camera overview pose.')
                continue
            with PICK_FILE.open('w', encoding='utf-8') as stream:
                json.dump(data, stream, indent=2)
            print(f'Pickup mapping saved. Fifth-point error {result.error * 1000:.1f} mm.')
            return
        if command not in ('1', '2', '3', '4', '5', 'D'):
            continue
        old = data['drop'] if command == 'D' else data['poses'][int(command) - 1]
        print(f'Record {command}; previous tool XYZ: {np.round(old[:3], 6).tolist()}')
        print('In Studio: lift clear, set the required orientation, then manually position the OPEN gripper.')
        print('At pickup, the die must be between the fingers without moving it. Keep the original pickup Z.')
        print('Use Pose with fixed Roll/Pitch/Yaw; inspect actual values after movement.')
        if input('When stopped at the corrected pose, type RECORD (anything else cancels): ').strip() != 'RECORD':
            continue
        try:
            updated = replace_pick_record(data, command, robot.get_pose().to_list())
        except ValueError as error:
            print(error)
            print('Previous record kept. No need to repeat the other targets.')
            continue
        save_pick_draft(updated)
        data = updated
        print(f'Record {command} saved. Lift before travelling to another target.')



def require_gripper_healthy(robot):
    status = robot.get_hardware_status()
    if not status.connection_up or status.calibration_needed or status.calibration_in_progress:
        raise ValueError('Robot hardware is disconnected or joint calibration is required/in progress.')
    problems = [f'{name}: code {code}' for name, code in
                zip(status.motor_names, status.hardware_errors) if code != 0]
    if problems:
        raise ValueError('Robot reports hardware errors: ' + ', '.join(problems)
                         + f'. Details: {status.error_message}')


class PickupPreflightError(ValueError):
    """Rejected before any actuator command; it is safe to keep detecting."""


def pick_and_place(robot, calibration, pixel, image_size, revalidate, workspace=None, box=None):
    """One confirmed cycle. Any failed command stops the remaining sequence."""
    require_pose(robot.get_pose().to_list(), calibration.overview, angle_tolerance=OVERVIEW_ANGLE_TOLERANCE)
    if gripper_name(robot) != calibration.tool:
        raise ValueError('Gripper changed. Re-teach pickup.')
    pickup = resolve_pick_target(calibration, pixel, image_size, workspace, box=box)
    above_pick = pickup.copy()
    above_pick[2] = calibration.overview[2]
    above_drop = calibration.drop.copy()
    above_drop[2] = calibration.overview[2]
    route = [above_pick, pickup, above_pick, above_drop, calibration.drop, above_drop, calibration.overview]
    print('Pick/place tool poses [metres, radians]:')
    for pose in route:
        print(np.round(pose, 5).tolist())
    print('Check whole arm route, gripper opening, neighbours and drop space. Speed remains at 10%.')
    if input('Type PICK to execute one cycle, anything else cancels: ').strip() != 'PICK':
        return False
    try:
        require_pose(robot.get_pose().to_list(), calibration.overview, angle_tolerance=OVERVIEW_ANGLE_TOLERANCE)
        if robot.get_learning_mode():
            raise ValueError('Disable learning mode before pickup.')
        if gripper_name(robot) != calibration.tool:
            raise ValueError('Gripper changed before pickup.')
        revalidate()
        require_gripper_healthy(robot)
    except ValueError as error:
        raise PickupPreflightError(str(error)) from error
    robot.set_arm_max_velocity(10)
    print('Gripper: speed 300, peak effort 30%, holding effort 10%.')

    def move(pose, upper=False, label='move'):
        print(f'Pick/place stage: {label}', flush=True)
        move_linear_pose_compatible(robot, pose)
        if upper:
            wait_for_pose(robot, pose, angle_tolerance=OVERVIEW_ANGLE_TOLERANCE)
        else:
            wait_for_pose(robot, pose)
        require_gripper_healthy(robot)

    robot.open_gripper(**GRIPPER_OPTIONS)
    require_gripper_healthy(robot)
    move(above_pick, upper=True, label='upper approach')
    move(pickup, label='descend to pickup')
    robot.close_gripper(**GRIPPER_OPTIONS)
    require_gripper_healthy(robot)
    move(above_pick, upper=True, label='lift after pickup')
    move(above_drop, upper=True, label='transfer above drop')
    move(calibration.drop, label='descend to drop')
    robot.open_gripper(**GRIPPER_OPTIONS)
    require_gripper_healthy(robot)
    move(above_drop, upper=True, label='lift after release')
    move(calibration.overview, upper=True, label='return to camera overview')
    return True



def wait_for_preview_pose(robot, pose):
    """Allow explicitly waiting again after delayed feedback, never resend motion."""
    while True:
        try:
            wait_for_pose(robot, pose, angle_tolerance=OVERVIEW_ANGLE_TOLERANCE)
            return
        except ValueError as error:
            # A hardware fault must not become an invitation to wait/retry.
            require_gripper_healthy(robot)
            print(f'Preview pose not confirmed: {error}')
            print('No new movement command will be sent. Keep clear of the arm.')
            if input('Type WAIT to check pose again, anything else stops here: ').strip() != 'WAIT':
                raise


def preview_pick(robot, calibration, pixel, image_size, revalidate, workspace=None, box=None, clearance_mm=None, speed_percent=50, inspection_callback=None):
    """Confirm an upper approach to any calibrated position, then return; no grip."""
    if isinstance(speed_percent, bool) or not isinstance(speed_percent, (int, np.integer)) or not 1 <= speed_percent <= 50:
        raise ValueError('Preview upper speed must be an integer from 1 to 50 percent.')
    if inspection_callback is not None and clearance_mm is None:
        raise ValueError('Live alignment requires a lower inspection height.')
    require_pose(robot.get_pose().to_list(), calibration.overview,
                 angle_tolerance=OVERVIEW_ANGLE_TOLERANCE)
    target = resolve_pick_target(calibration, pixel, image_size, workspace, box=box)
    above = target.copy()
    above[2] = calibration.overview[2]
    if above[2] <= target[2] + .002:
        raise ValueError('Overview must be above the taught pickup height.')
    lower = None
    if clearance_mm is not None:
        if not np.isfinite(clearance_mm) or clearance_mm < 30:
            raise ValueError('Preview clearance must be at least 30 mm ABOVE taught grip height.')
        lower = target.copy()
        lower[2] = target[2] + clearance_mm / 1000
        if lower[2] >= above[2] - .002:
            raise ValueError('Preview inspection height must be below the overview.')
        print('LOW PREVIEW: no gripper commands. Inspect the WHOLE arm route before confirming.')
        print('Inspection pose:', np.round(lower,6).tolist())
        print('Height is relative to taught tool grip Z, not a measured clearance from the table.')
    else:
        print('PREVIEW ONLY: upper approach and return. No descent or gripper commands.')
    print('Calculated pickup XYZ/RPY (NOT executed):', np.round(target, 6).tolist())
    print('Executed upper pose:', np.round(above, 6).tolist())
    print(f'Upper approach speed: {speed_percent}%. Lower inspection, rise and return: 20%. Check the whole arm path.')
    if input('Type PREVIEW to execute, anything else cancels: ').strip() != 'PREVIEW':
        return False
    try:
        require_pose(robot.get_pose().to_list(), calibration.overview,
                     angle_tolerance=OVERVIEW_ANGLE_TOLERANCE)
        if robot.get_learning_mode():
            raise ValueError('Disable Learning Mode first.')
        if gripper_name(robot) != calibration.tool:
            raise ValueError('Configured gripper changed.')
        revalidate()
        require_gripper_healthy(robot)
    except ValueError as error:
        raise PickupPreflightError(str(error)) from error
    robot.set_arm_max_velocity(speed_percent)
    current_speed = speed_percent
    route = [('upper preview', above)]
    if lower is not None:
        route.extend([('lower inspection', lower), ('rise after inspection', above)])
    route.append(('return to overview', calibration.overview))
    for label, pose in route:
        stage_speed = speed_percent if label == 'upper preview' else 20
        if stage_speed != current_speed:
            robot.set_arm_max_velocity(stage_speed)
            current_speed = stage_speed
        print(f'Preview stage: {label}', flush=True)
        move_linear_pose_compatible(robot, pose)
        wait_for_preview_pose(robot, pose)
        require_gripper_healthy(robot)
        if label == 'lower inspection':
            if inspection_callback is not None:
                corrected = pose_array(inspection_callback(lower.copy()))
                if not np.allclose(corrected[2:], lower[2:], atol=1e-9, rtol=0):
                    raise ValueError('Invalid inspection correction. Stopped without recovery movement.')
                require_pose(robot.get_pose().to_list(), corrected, angle_tolerance=OVERVIEW_ANGLE_TOLERANCE)
                above[:2] = corrected[:2]  # Rise vertically from the corrected XY.
                robot.set_arm_max_velocity(20)
                current_speed = 20
            elif input('Inspect alignment on the REAL robot. Type RETURN to rise and return; anything else stops here: ').strip() != 'RETURN':
                raise ValueError('Stopped at inspection height. No automatic recovery movement.')
    if current_speed != 10:
        robot.set_arm_max_velocity(10)
    return True



def alignment_step(origin, current, axis, direction, step_mm=5):
    """An explicit operator step in ROBOT axes; keep height and orientation."""
    if axis not in (0, 1) or direction not in (-1, 1):
        raise ValueError('Only robot X/Y steps are allowed.')
    if isinstance(step_mm, bool) or step_mm not in (1, 2, 5):
        raise ValueError('Alignment step must be 1, 2 or 5 mm.')
    origin, current = pose_array(origin), pose_array(current)
    if not np.allclose(current[2:], origin[2:], atol=1e-9, rtol=0):
        raise ValueError('Alignment must keep Z and orientation fixed.')
    target = current.copy()
    target[axis] += direction * step_mm / 1000
    return target


def live_inspection(robot, planned, intrinsics, robot_ip):
    """Operator-guided camera inspection. No automatic tracking, descent or grip."""
    import time
    origin = pose_array(planned)
    current = origin.copy()
    title = 'LIVE MANUAL ALIGNMENT - robot axes, NOT image axes'
    keys = {ord('a'): (0,-1), ord('d'): (0,1), ord('s'): (1,-1), ord('w'): (1,1)}
    require_live_camera(robot_ip)
    require_gripper_healthy(robot)
    initial_actual = pose_array(robot.get_pose().to_list())
    require_pose(initial_actual, current, angle_tolerance=OVERVIEW_ANGLE_TOLERANCE)
    print('Live MANUAL alignment: A/D = robot X -/+, S/W = robot Y -/+. Default step 5 mm.')
    print('Keys 1/2/3 select steps 1/2/5 mm. Speed 20%. Release keys between taps.')
    print('Watch the REAL fingers. Z and rotation stay fixed. No total XY displacement limit.')
    print('Enter = save measurement, rise and return. Q/Esc or window close = STOP in place.')
    robot.set_arm_max_velocity(20)
    step_mm = 5
    armed = True
    released_at = None
    checked_at = time.monotonic()
    cv2.namedWindow(title)
    try:
        while True:
            if time.monotonic() - checked_at > 2:
                require_live_camera(robot_ip)
                require_gripper_healthy(robot)
                checked_at = time.monotonic()
            frame = read_frame(robot, intrinsics)
            delta = (current[:2]-origin[:2])*1000
            for index, line in enumerate([
                f'MANUAL: A/D robot X -/+; S/W robot Y -/+; step {step_mm} mm',
                f'dX {delta[0]:+.1f} mm  dY {delta[1]:+.1f} mm; Z FIXED; speed 20%',
                'ENTER: save + rise/return | Q/ESC: STOP here, no recovery',
                '1/2/3: step 1/2/5 mm. Robot axes are NOT camera left/right.']):
                cv2.putText(frame,line,(8,22+24*index),cv2.FONT_HERSHEY_SIMPLEX,.46,(0,0,255),1)
            cv2.imshow(title,frame)
            key = cv2.waitKey(30)
            if cv2.getWindowProperty(title,cv2.WND_PROP_VISIBLE) < 1 or key in (27,ord('q')):
                raise ValueError('Live alignment stopped in place. No automatic recovery movement.')
            if key == -1:
                if released_at is None:
                    released_at = time.monotonic()
                if time.monotonic()-released_at >= .3:
                    armed = True
                continue
            released_at = None
            if not armed:
                continue
            armed = False
            if key in (ord('1'),ord('2'),ord('3')):
                step_mm = {ord('1'):1,ord('2'):2,ord('3'):5}[key]
                continue
            if key not in keys and key not in (10,13):
                continue
            require_live_camera(robot_ip)
            checked_at = time.monotonic()
            require_gripper_healthy(robot)
            if robot.get_learning_mode():
                raise ValueError('Learning Mode enabled. Stopped without recovery movement.')
            require_pose(robot.get_pose().to_list(),current,angle_tolerance=OVERVIEW_ANGLE_TOLERANCE)
            if key in (10,13):
                report = dict(mode='manual_preview_only', planned=origin.tolist(),
                              initial_actual=initial_actual.tolist(), corrected_command=current.tolist(),
                              corrected_actual=robot.get_pose().to_list(),
                              delta_xy_mm=delta.tolist(), step_mm=step_mm, speed_percent=20,
                              automatic_tracking=False, automatically_applied=False)
                path = Path(__file__).with_name('dice_alignment_measurement.json')
                temporary = path.with_suffix('.json.tmp')
                temporary.write_text(json.dumps(report,indent=2),encoding='utf-8')
                temporary.replace(path)
                print(f'Alignment measurement saved: {path}. Pickup calibration unchanged.')
                return current
            try:
                target = alignment_step(origin,current,*keys[key],step_mm=step_mm)
            except ValueError as error:
                print(error)
                continue
            move_linear_pose_compatible(robot,target)
            wait_for_preview_pose(robot,target)
            require_gripper_healthy(robot)
            current = target
    finally:
        cv2.destroyWindow(title)
