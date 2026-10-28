import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import cv2
import numpy as np
from pyniryo import NiryoRobot

from dice_camera_calibration import read_frame
from dice_workspace import detect_workspace


ROBOT_IP = "169.254.200.200"
OUTPUT = Path(__file__).with_name("dice_hand_eye_calibration.json")
SAMPLES = Path(__file__).with_name("dice_hand_eye_samples.json")
MIN_SAMPLES = 8
POSITION_LIMIT = 0.005
ANGLE_LIMIT = np.deg2rad(2)
JOINT6_LIMIT = 0.01


def transform(rotation, translation):
    matrix = np.eye(4)
    matrix[:3, :3] = rotation
    matrix[:3, 3] = np.asarray(translation).reshape(3)
    return matrix


def pose_transform(pose):
    values = np.asarray(pose, dtype=float)
    if values.shape != (6,) or not np.isfinite(values).all():
        raise ValueError("Invalid robot XYZ/RPY pose.")
    roll, pitch, yaw = values[3:]
    rx = cv2.Rodrigues(np.array([roll, 0, 0]))[0]
    ry = cv2.Rodrigues(np.array([0, pitch, 0]))[0]
    rz = cv2.Rodrigues(np.array([0, 0, yaw]))[0]
    return transform(rz @ ry @ rx, values[:3])


def rotation_angle(rotation):
    return float(np.arccos(np.clip((np.trace(rotation) - 1) / 2, -1, 1)))


def require_transform(value):
    matrix = np.asarray(value, dtype=float)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError("Expected a finite 4x4 rigid transform.")
    rotation = matrix[:3, :3]
    if (
        not np.allclose(matrix[3], [0, 0, 0, 1])
        or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6)
        or not np.isclose(np.linalg.det(rotation), 1, atol=1e-6)
    ):
        raise ValueError("Invalid rigid transform rotation or homogeneous row.")
    return matrix


def board_points(size_m):
    if not np.isfinite(size_m) or not 0 < size_m < 1:
        raise ValueError("Board side must be positive and less than one metre.")
    return np.array([[0, 0, 0], [size_m, 0, 0],
                     [size_m, size_m, 0], [0, size_m, 0]], dtype=float)


def estimate_board_pose(corners, camera_matrix, size_m):
    points = board_points(size_m)
    corners = np.asarray(corners, dtype=float)
    camera_matrix = np.asarray(camera_matrix, dtype=float)
    if corners.shape != (4, 2) or not np.isfinite(corners).all():
        raise ValueError("Expected four finite marker centres.")
    if (
        camera_matrix.shape != (3, 3) or not np.isfinite(camera_matrix).all()
        or camera_matrix[0, 0] <= 0 or camera_matrix[1, 1] <= 0
    ):
        raise ValueError("Invalid camera intrinsics.")
    count, rotations, translations, _ = cv2.solvePnPGeneric(
        points, corners, camera_matrix, np.zeros(5), flags=cv2.SOLVEPNP_IPPE
    )
    if not count:
        raise ValueError("No planar board pose found.")
    candidates = []
    for rvec, tvec in zip(rotations, translations):
        rotation = cv2.Rodrigues(rvec)[0]
        pose = transform(rotation, tvec)
        if not np.isfinite(pose).all() or np.any((points @ rotation.T + tvec.ravel())[:, 2] <= 0):
            continue
        projected = cv2.projectPoints(points, rvec, tvec, camera_matrix, np.zeros(5))[0]
        error = float(np.sqrt(np.mean(np.sum((projected.reshape(-1, 2) - corners) ** 2, axis=1))))
        candidates.append((error, pose))
    candidates.sort(key=lambda item: item[0])
    if not candidates or candidates[0][0] > 3:
        raise ValueError("Board reprojection error exceeds 3 pixels; improve focus/visibility.")
    if len(candidates) > 1 and candidates[1][0] - candidates[0][0] < 0.5:
        difference = np.linalg.inv(candidates[0][1]) @ candidates[1][1]
        if rotation_angle(difference[:3, :3]) > np.deg2rad(5):
            raise ValueError("Ambiguous planar pose; use another safely tilted overview.")
    return candidates[0][1], candidates[0][0]


def solve_hand_eye(base_tools, camera_boards):
    """Solve A X = X B with rotation-vector SVD, then translation least squares."""
    motions = []
    for i in range(len(base_tools)):
        for j in range(i):
            a = np.linalg.inv(base_tools[j]) @ base_tools[i]
            b = camera_boards[j] @ np.linalg.inv(camera_boards[i])
            if rotation_angle(a[:3, :3]) > np.deg2rad(2):
                motions.append((a, b))
    if len(motions) < 3:
        raise ValueError("Too little rotation: collect different overview orientations.")
    a_vectors = np.array([cv2.Rodrigues(a[:3, :3])[0].ravel() for a, _ in motions])
    b_vectors = np.array([cv2.Rodrigues(b[:3, :3])[0].ravel() for _, b in motions])
    singular = np.linalg.svd(a_vectors, compute_uv=False)
    if (
        np.max(np.linalg.norm(a_vectors, axis=1)) < np.deg2rad(15)
        or singular[1] < 0.1 or singular[1] / singular[0] < 0.15
    ):
        raise ValueError("Need rotations about TWO different axes, not just translations or yaw.")
    u, _, vt = np.linalg.svd(a_vectors.T @ b_vectors)
    rotation = u @ np.diag([1, 1, np.linalg.det(u @ vt)]) @ vt
    lhs = np.vstack([a[:3, :3] - np.eye(3) for a, _ in motions])
    rhs = np.concatenate([rotation @ b[:3, 3] - a[:3, 3] for a, b in motions])
    translation, _, rank, _ = np.linalg.lstsq(lhs, rhs, rcond=None)
    if rank != 3 or not np.isfinite(translation).all() or np.linalg.norm(translation) > 0.5:
        raise ValueError("Degenerate or implausible camera-to-tool translation.")
    return require_transform(transform(rotation, translation))


def fit_samples(samples, size_m):
    if len(samples) < MIN_SAMPLES:
        raise ValueError("Need at least 8 views: 6 training views plus 2 independent checks.")
    base_tools = [require_transform(sample["base_tool"]) for sample in samples]
    camera_boards = [require_transform(sample["camera_board"]) for sample in samples]
    joints = np.array([sample["joints"] for sample in samples], dtype=float)
    if joints.shape != (len(samples), 6) or not np.isfinite(joints).all():
        raise ValueError("Each view must contain six finite joint angles.")
    if np.max(np.abs(joints[:, 5] - joints[0, 5])) > JOINT6_LIMIT:
        raise ValueError("Joint 6 changed. Keep it fixed during capture and future use.")
    train_count = len(samples) - 2
    tool_camera = solve_hand_eye(base_tools[:train_count], camera_boards[:train_count])
    board_poses = [g @ tool_camera @ c for g, c in zip(base_tools, camera_boards)]
    u, _, vt = np.linalg.svd(sum(pose[:3, :3] for pose in board_poses[:train_count]))
    mean_rotation = u @ np.diag([1, 1, np.linalg.det(u @ vt)]) @ vt
    base_board = transform(
        mean_rotation, np.mean([pose[:3, 3] for pose in board_poses[:train_count]], axis=0)
    )
    corners = board_points(size_m)
    reference = corners @ base_board[:3, :3].T + base_board[:3, 3]
    errors = []
    for index, pose in enumerate(board_poses):
        actual = corners @ pose[:3, :3].T + pose[:3, 3]
        distance = float(np.max(np.linalg.norm(actual - reference, axis=1)))
        angle = rotation_angle(base_board[:3, :3].T @ pose[:3, :3])
        errors.append({"max_corner_error_mm": distance * 1000,
                       "angle_error_deg": float(np.rad2deg(angle))})
        if distance > POSITION_LIMIT or angle > ANGLE_LIMIT:
            raise ValueError(
                f"View {index + 1} inconsistent: {distance * 1000:.1f} mm, "
                f"{np.rad2deg(angle):.1f} deg. Check board movement, corner labels, "
                "camera rigidity and pose diversity. Calibration NOT saved."
            )
    return {
        "version": 1,
        "status": "geometry_checked_not_motion_approved",
        "motion_enabled": False,
        "units": "metres",
        "board_side_m": size_m,
        "tool_from_camera": tool_camera.tolist(),
        "base_from_board": base_board.tolist(),
        "required_joint6_rad": float(joints[0, 5]),
        "training_views": train_count,
        "validation_views": 2,
        "view_errors": errors,
    }


def capture_sample(robot, intrinsics, size_m, samples):
    before = pose_transform(robot.get_pose().to_list())
    joints_before = np.asarray(robot.get_joints(), dtype=float)
    first_frame = read_frame(robot, intrinsics)
    first = detect_workspace(first_frame)
    time.sleep(0.25)
    frame = read_frame(robot, intrinsics)
    workspace = detect_workspace(frame)
    after = pose_transform(robot.get_pose().to_list())
    joints = np.asarray(robot.get_joints(), dtype=float)
    delta = np.linalg.inv(before) @ after
    if (
        np.linalg.norm(delta[:3, 3]) > 0.0005
        or rotation_angle(delta[:3, :3]) > 0.002
        or joints.shape != (6,) or not np.isfinite(joints).all()
        or joints_before.shape != (6,) or not np.isfinite(joints_before).all()
        or np.max(np.abs(joints - joints_before)) > 0.002
    ):
        raise ValueError("Robot is moving. Stop manual control, wait, then capture again.")
    if first is None or workspace is None:
        raise ValueError("All four ring markers must be visible in both capture frames.")
    if np.max(np.linalg.norm(first.corners - workspace.corners, axis=1)) > 1.5:
        raise ValueError("Image is not stable. Wait for the camera, then capture again.")
    if samples:
        if list(frame.shape[1::-1]) != samples[0]["image_size"]:
            raise ValueError("Camera resolution changed; restart capture.")
        if abs(joints[5] - samples[0]["joints"][5]) > JOINT6_LIMIT:
            raise ValueError("Joint 6 changed. Restore its original value before capturing.")
        for sample in samples:
            relative = np.linalg.inv(np.asarray(sample["base_tool"])) @ after
            if (
                np.linalg.norm(relative[:3, 3]) < 0.002
                and rotation_angle(relative[:3, :3]) < np.deg2rad(2)
            ):
                raise ValueError("Duplicate view. Use a different safe overview pose.")
    camera_board, error = estimate_board_pose(workspace.corners, intrinsics[0], size_m)
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "base_tool": after.tolist(),
        "camera_board": camera_board.tolist(),
        "joints": joints.tolist(),
        "corners": workspace.corners.tolist(),
        "image_size": list(frame.shape[1::-1]),
        "reprojection_rmse_px": error,
    }


def write_json(path, data):
    temporary = path.with_suffix(".json.tmp")
    try:
        temporary.write_text(json.dumps(data, indent=2, allow_nan=False), encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description="Read-only non-contact hand-eye calibration.")
    parser.add_argument("--board-size-mm", type=float, default=170,
                        help="Measured marker-centre square side in mm (default: 170).")
    args = parser.parse_args()
    size_m = args.board_size_mm / 1000
    board_points(size_m)
    print(
        f"READ-ONLY hand-eye capture. Marker-centre square: {args.board_size_mm:g} mm.\n"
        "No script-controlled motion. Use Studio for safe overview poses ONLY.\n"
        "Keep board, camera mounting and tool fixed. Keep joint 6 at its original angle.\n"
        "Keep M1-M4 on the SAME physical markers; avoid large camera roll.\n"
        "Collect 6+ diverse views with rotations about TWO axes; last 2 views are checks.\n"
        "Stop manual control and wait at least one second before C.\n"
        "C = capture | U = undo | F = fit/check/save | Q = quit.\n"
        "Do not descend towards or touch the board. No movement is enabled by the result."
    )
    if input("Type CAPTURE to start a new session, anything else cancels: ").strip() != "CAPTURE":
        return
    robot = None
    samples = []
    try:
        robot = NiryoRobot(ROBOT_IP)
        status = robot.get_hardware_status()
        if status.calibration_needed or status.calibration_in_progress:
            raise ValueError("Finish joint calibration in Studio before recording.")
        intrinsics = robot.get_camera_intrinsics()
        initial_joint6 = robot.get_joints()[5]
        metadata = {
            "hardware_version": status.hardware_version,
            "board_side_m": size_m,
            "camera_matrix": np.asarray(intrinsics[0]).tolist(),
            "distortion": np.asarray(intrinsics[1]).tolist(),
            "initial_joint6_rad": initial_joint6,
        }
        write_json(SAMPLES, {**metadata, "status": "unvalidated", "samples": samples})
        print("Joint 6 initial angle:", initial_joint6, "rad. Keep this value.")
        while True:
            frame = read_frame(robot, intrinsics)
            workspace = detect_workspace(frame)
            if workspace is not None:
                cv2.polylines(frame, [workspace.corners], True, (255, 255, 0), 2)
                for index, corner in enumerate(workspace.corners, 1):
                    point = tuple(map(int, corner))
                    cv2.putText(frame, f"M{index}", point,
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            label = f"READ ONLY | views: {len(samples)} | C capture, U undo, F finish, Q quit"
            cv2.putText(frame, label, (8, frame.shape[0] - 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
            cv2.imshow("Hand-eye calibration - no automatic motion", frame)
            key = cv2.waitKey(20) & 0xFF
            if key == ord("q"):
                print(f"Stopped. Unvalidated observations remain at {SAMPLES}")
                break
            if key == ord("c"):
                try:
                    sample = capture_sample(robot, intrinsics, size_m, samples)
                    if abs(sample["joints"][5] - initial_joint6) > JOINT6_LIMIT:
                        raise ValueError("Restore joint 6 to its session-start angle.")
                except (ValueError, cv2.error) as error:
                    print(f"Capture rejected: {error}")
                    continue
                samples.append(sample)
                print(f"View {len(samples)} captured; reprojection RMSE "
                      f"{sample['reprojection_rmse_px']:.2f} px.")
            elif key == ord("u"):
                if samples:
                    samples.pop()
                    print(f"Removed last view. Remaining: {len(samples)}")
                else:
                    print("No views to remove.")
            elif key == ord("f"):
                try:
                    result = fit_samples(samples, size_m)
                except (ValueError, cv2.error, np.linalg.LinAlgError) as error:
                    print(f"Calibration rejected: {error}")
                    continue
                write_json(OUTPUT, {**metadata, **result, "samples": samples})
                print(f"Geometry checks passed. Saved: {OUTPUT}")
                print("This result does NOT enable motion. Independent physical validation "
                      "and a collision-safe approach are still required.")
                break
            if key in (ord("c"), ord("u")):
                write_json(SAMPLES, {**metadata, "status": "unvalidated", "samples": samples})
    finally:
        cv2.destroyAllWindows()
        if robot is not None:
            robot.close_connection()


if __name__ == "__main__":
    main()
