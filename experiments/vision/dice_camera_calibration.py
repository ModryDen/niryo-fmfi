import json
from pathlib import Path

import cv2
import numpy as np


CALIBRATION_FILE = Path(__file__).with_name("dice_camera_calibration.json")
POSITION_TOLERANCE = 0.002
ANGLE_TOLERANCE = 0.02
OVERVIEW_ANGLE_TOLERANCE = np.deg2rad(2.0)
VALIDATION_TOLERANCE = 0.005


def read_frame(robot, intrinsics):
    frame = cv2.imdecode(
        np.frombuffer(robot.get_img_compressed(), dtype=np.uint8),
        cv2.IMREAD_COLOR,
    )
    if frame is None:
        raise ValueError("Cannot decode the robot camera image.")
    matrix, distortion = intrinsics
    return cv2.undistort(frame, matrix, distortion)



def require_live_camera(ip_address, timeout=3.0):
    """Read ROS metadata: active stream and two distinct newly received frames.

    TCP GET_IMAGE can return a cached JPEG even when streaming is off.
    This function sends no camera control or robot motion commands.
    """
    import json
    import time
    import websocket
    connection = None
    try:
        connection = websocket.create_connection(f'ws://{ip_address}:9090', timeout=timeout)
        active_topic = '/niryo_robot_vision/video_stream_is_active'
        video_topic = '/niryo_robot_vision/compressed_video_stream'
        for topic in (active_topic, video_topic):
            connection.send(json.dumps(dict(op='subscribe', id='pick-live-'+topic,
                                           topic=topic, queue_length=1, throttle_rate=200)))
        deadline = time.monotonic() + timeout
        active = False
        frames = set()
        while time.monotonic() < deadline:
            connection.settimeout(max(.01, deadline-time.monotonic()))
            packet = json.loads(connection.recv())
            if packet.get('topic') == active_topic:
                active = packet.get('msg', {}).get('data') is True
                if not active:
                    raise ValueError('Camera stream is OFF. The displayed JPEG may be an old cached image. '
                                     'Restore live video before picking; do not move dice to the old cross.')
            if packet.get('topic') == video_topic:
                header = packet.get('msg', {}).get('header', {})
                stamp = header.get('stamp', {})
                identity = (header.get('seq'), stamp.get('secs'), stamp.get('nsecs'))
                if any(value not in (None, 0) for value in identity):
                    frames.add(identity)
            if active and len(frames) >= 2:
                return
        raise ValueError('No fresh camera frames confirmed. Picking blocked.')
    except ValueError:
        raise
    except Exception as error:
        raise ValueError(f'Cannot confirm live camera stream: {error}. Picking blocked.') from error
    finally:
        if connection is not None:
            connection.close()


def pose_array(pose):
    values = np.asarray(pose, dtype=float)
    if values.shape != (6,) or not np.isfinite(values).all():
        raise ValueError("Expected a finite XYZ/RPY pose in metres/radians.")
    return values


def orientation_distance(first, second):
    """Physical rotation angle for XYZ/RPY poses, including near pitch +/- pi/2."""
    def rotation(pose):
        roll, pitch, yaw = pose_array(pose)[3:]
        cr, cp, cy = np.cos([roll, pitch, yaw])
        sr, sp, sy = np.sin([roll, pitch, yaw])
        return np.array([
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ])
    relative = rotation(first).T @ rotation(second)
    return float(np.arccos(np.clip((np.trace(relative) - 1) / 2, -1, 1)))



def move_linear_pose_compatible(robot, pose):
    """Handle the installed SDK's extra frame argument on six-argument servers.

    Retry only an explicit argument-count rejection, never a motion/network error.
    The legacy path retains MOVE_LINEAR_POSE and the same six base-frame values.
    """
    from pyniryo.api.enums_communication import Command
    from pyniryo.api.exceptions import NiryoRobotException
    values = pose_array(pose).tolist()
    sender = getattr(robot, '_NiryoRobot__send_n_receive', None)
    if getattr(robot, '_dice_legacy_linear_pose', False) is True:
        if not callable(sender):
            raise ValueError('Legacy Niryo protocol sender is unavailable.')
        return sender(Command.MOVE_LINEAR_POSE, *values)
    try:
        return robot.move_linear_pose(values)
    except NiryoRobotException as error:
        if '6 parameters expected but 7 given' not in str(error) or not callable(sender):
            raise
        # The server rejected the parameter list before executing the command.
        print('Niryo compatibility: using six-parameter linear pose commands.')
        result = sender(Command.MOVE_LINEAR_POSE, *values)
        robot._dice_legacy_linear_pose = True
        return result


def same_orientation(first, second):
    return orientation_distance(first, second) <= ANGLE_TOLERANCE


def require_pose(actual, expected, angle_tolerance=ANGLE_TOLERANCE):
    actual, expected = pose_array(actual), pose_array(expected)
    if (
        np.linalg.norm(actual[:3] - expected[:3]) > POSITION_TOLERANCE
        or orientation_distance(actual, expected) > angle_tolerance
    ):
        raise ValueError(
            "Robot is not at the required taught pose. "
            f"Position difference: {np.linalg.norm(actual[:3] - expected[:3]) * 1000:.2f} mm "
            f"(limit {POSITION_TOLERANCE * 1000:.1f}); "
            f"orientation difference: {np.degrees(orientation_distance(actual, expected)):.2f} deg "
            f"(limit {np.degrees(angle_tolerance):.2f}). "
            f"Actual XYZ/RPY: {np.round(actual, 6).tolist()}; "
            f"required: {np.round(expected, 6).tolist()}."
        )



def wait_for_pose(robot, expected, timeout=3.0, interval=.1, angle_tolerance=ANGLE_TOLERANCE):
    """Wait for two matching readings after a command; never resend motion."""
    import time
    expected = pose_array(expected)
    deadline = time.monotonic() + timeout
    matching = 0
    last_error = None
    while True:
        try:
            require_pose(robot.get_pose().to_list(), expected, angle_tolerance=angle_tolerance)
            matching += 1
            if matching >= 2:
                return
        except ValueError as error:
            matching = 0
            last_error = error
        if time.monotonic() >= deadline:
            if last_error is not None:
                raise ValueError(f'Pose did not settle within {timeout:.1f}s: {last_error}')
            raise ValueError('Pose stability was not confirmed before timeout.')
        time.sleep(interval)


def fit_calibration(pixels, poses, overview, image_size):
    pixels = np.asarray(pixels, dtype=np.float32)
    poses = np.asarray([pose_array(pose) for pose in poses])
    overview = pose_array(overview)
    if pixels.shape != (5, 2) or poses.shape != (5, 6):
        raise ValueError("Teach four boundary points and one independent check point.")
    if not np.isfinite(pixels).all():
        raise ValueError("Calibration pixels must be finite.")
    width, height = image_size
    if np.any(pixels < 0) or np.any(pixels >= [width, height]):
        raise ValueError("Calibration points must lie inside the camera image.")
    boundary = cv2.convexHull(pixels[:4]).reshape(-1, 2)
    if len(boundary) != 4 or cv2.contourArea(boundary) < 2500:
        raise ValueError("The first four points must span a convex board area.")
    if cv2.pointPolygonTest(boundary, tuple(map(float, pixels[4])), False) <= 0:
        raise ValueError("The fifth check point must be inside the first four.")
    if not all(same_orientation(overview, pose) for pose in poses):
        raise ValueError("Keep the SAME camera orientation as the overview at all points.")
    if np.ptp(poses[:, 2]) > POSITION_TOLERANCE:
        raise ValueError("Use the SAME safe close-up Z height at all five points.")
    if overview[2] <= np.max(poses[:, 2]) + POSITION_TOLERANCE:
        raise ValueError("The overview must be above the close-up height.")
    xy_boundary = cv2.convexHull(poses[:4, :2].astype(np.float32))
    if len(xy_boundary) != 4 or cv2.contourArea(xy_boundary) < 0.0001:
        raise ValueError("Taught camera positions must span a convex board area.")
    # Empirical camera-centred poses include the camera-to-tool offset.
    matrix = cv2.getPerspectiveTransform(
        pixels[:4], poses[:4, :2].astype(np.float32)
    )
    if not np.isfinite(matrix).all() or np.linalg.matrix_rank(matrix) != 3:
        raise ValueError("Degenerate calibration.")
    denominators = np.column_stack((pixels[:4], np.ones(4))) @ matrix[2]
    if not (np.all(denominators > 1e-8) or np.all(denominators < -1e-8)):
        raise ValueError("Calibration has a projection singularity inside the board.")
    predicted = cv2.perspectiveTransform(pixels[4:].reshape(1, 1, 2), matrix)[0, 0]
    error = float(np.linalg.norm(predicted - poses[4, :2]))
    if not np.isfinite(error) or error > VALIDATION_TOLERANCE:
        raise ValueError(f"Independent check error {error * 1000:.1f} mm exceeds 5 mm.")
    return matrix, boundary, error


class CameraCalibration:
    def __init__(self, data):
        if data.get("version") != 1:
            raise ValueError("Unsupported camera calibration format.")
        self.image_size = tuple(data["image_size"])
        self.overview = pose_array(data["overview"])
        self.poses = [pose_array(pose) for pose in data["poses"]]
        self.matrix, self.boundary, self.error = fit_calibration(
            data["pixels"], self.poses, self.overview, self.image_size
        )
        self.intrinsics = (
            np.asarray(data["camera_matrix"], dtype=float),
            np.asarray(data["distortion"], dtype=float),
        )

    def target(self, pixel, image_size):
        if tuple(image_size) != self.image_size:
            raise ValueError("Camera resolution changed. Repeat calibration.")
        point = tuple(map(float, pixel))
        if not np.isfinite(point).all():
            raise ValueError("Target pixels must be finite.")
        if cv2.pointPolygonTest(self.boundary, point, False) < 0:
            raise ValueError("Selected die is outside the calibrated area.")
        xy = cv2.perspectiveTransform(
            np.asarray([[point]], dtype=np.float32), self.matrix
        )[0, 0]
        target = self.overview.copy()
        target[:2] = xy
        target[2] = float(np.mean([pose[2] for pose in self.poses]))
        return pose_array(target)


def load_calibration(intrinsics):
    with CALIBRATION_FILE.open(encoding="utf-8") as stream:
        calibration = CameraCalibration(json.load(stream))
    for actual, saved in zip(intrinsics, calibration.intrinsics):
        if np.shape(actual) != saved.shape or not np.allclose(actual, saved):
            raise ValueError("Camera intrinsics changed. Repeat calibration.")
    return calibration


def teach_calibration(robot, intrinsics):
    print(
        "NO automatic motion during teaching. Use Niryo Studio to move the robot.\n"
        "Start at the saved overhead pose. Keep XYZ orientation unchanged throughout.\n"
        "Use five stationary dice/targets of the SAME height as the future dice.\n"
        "Teach a safe close-up Z with adequate clearance and only one die in view.\n"
        "The raised horizontal route and all vertical routes must be unobstructed.\n"
        "First select four outer target centres, then an interior check target."
    )
    if input("Type CALIBRATE to begin, anything else cancels: ").strip() != "CALIBRATE":
        return
    overview = robot.get_pose().to_list()
    reference = read_frame(robot, intrinsics)
    points = []
    window = "Calibration - click 4 outer centres, then 1 interior centre"

    def click(event, x, y, _flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 5:
            points.append((x, y))

    cv2.namedWindow(window)
    cv2.setMouseCallback(window, click)
    while True:
        annotated = reference.copy()
        for index, point in enumerate(points, 1):
            cv2.circle(annotated, point, 5, (0, 0, 255), -1)
            cv2.putText(annotated, str(index), point, cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 255, 0), 2)
        cv2.imshow(window, annotated)
        key = cv2.waitKey(20) & 0xFF
        if key == ord("q"):
            return
        if key == ord("r"):
            points.clear()
        if len(points) == 5 and key in (10, 13):
            break

    poses = []
    for index in range(5):
        print(
            f"Target {index + 1} at overview pixel {points[index]}: manually centre it "
            "on the live crosshair at the SAME safe close-up Z and orientation. "
            "Release manual control, then press C to capture; Q cancels."
        )
        while True:
            frame = read_frame(robot, intrinsics)
            height, width = frame.shape[:2]
            cv2.drawMarker(frame, (width // 2, height // 2), (0, 0, 255),
                           cv2.MARKER_CROSS, 30, 2)
            cv2.imshow("Calibration - live close-up", frame)
            key = cv2.waitKey(20) & 0xFF
            if key == ord("q"):
                return
            if key == ord("c"):
                poses.append(robot.get_pose().to_list())
                break
    data = {
        "version": 1,
        "image_size": [reference.shape[1], reference.shape[0]],
        "overview": overview,
        "pixels": points,
        "poses": poses,
        "camera_matrix": np.asarray(intrinsics[0]).tolist(),
        "distortion": np.asarray(intrinsics[1]).tolist(),
    }
    calibration = CameraCalibration(data)
    with CALIBRATION_FILE.open("w", encoding="utf-8") as stream:
        json.dump(data, stream, indent=2)
    print(f"Calibration saved: {CALIBRATION_FILE}")
    print(f"Independent check error: {calibration.error * 1000:.1f} mm.")
    print("Manually return to the ORIGINAL overview pose before normal detection.")


def confirmed_move(robot, expected, waypoints):
    require_pose(robot.get_pose().to_list(), expected)
    print("Planned linear tool poses [x, y, z, roll, pitch, yaw]:")
    for waypoint in waypoints:
        print(np.round(waypoint, 5).tolist())
    print(
        "Check the ENTIRE arm/tool path is clear, dice/board have not moved, "
        "and emergency stop is accessible. Speed will be set to 10% (not restored)."
    )
    if input("Type MOVE to confirm, anything else cancels: ").strip() != "MOVE":
        print("Motion cancelled.")
        return False
    require_pose(robot.get_pose().to_list(), expected)
    if robot.get_learning_mode():
        raise ValueError("Disable learning mode in Niryo Studio before confirming motion.")
    robot.set_arm_max_velocity(10)
    for waypoint in waypoints:
        move_linear_pose_compatible(robot, waypoint)
        wait_for_pose(robot, waypoint)
    return True
