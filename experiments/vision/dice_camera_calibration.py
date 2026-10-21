import json
from pathlib import Path

import cv2
import numpy as np


CALIBRATION_FILE = Path(__file__).with_name("dice_camera_calibration.json")
POSITION_TOLERANCE = 0.002
ANGLE_TOLERANCE = 0.02
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


def pose_array(pose):
    values = np.asarray(pose, dtype=float)
    if values.shape != (6,) or not np.isfinite(values).all():
        raise ValueError("Expected a finite XYZ/RPY pose in metres/radians.")
    return values


def same_orientation(first, second):
    difference = (first[3:] - second[3:] + np.pi) % (2 * np.pi) - np.pi
    return bool(np.all(np.abs(difference) <= ANGLE_TOLERANCE))


def require_pose(actual, expected):
    actual, expected = pose_array(actual), pose_array(expected)
    if (
        np.linalg.norm(actual[:3] - expected[:3]) > POSITION_TOLERANCE
        or not same_orientation(actual, expected)
    ):
        raise ValueError("Robot is not at the required taught pose. Move it manually first.")


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
        robot.move_linear_pose(pose_array(waypoint).tolist())
        require_pose(robot.get_pose().to_list(), waypoint)
    return True
