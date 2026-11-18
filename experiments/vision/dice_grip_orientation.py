"""Image-edge to robot yaw alignment with an explicitly taught jaw reference."""
import json
from pathlib import Path
import cv2
import numpy as np

REFERENCE_FILE = Path(__file__).with_name('dice_grip_orientation.json')


def rotation(rpy):
    r, p, y = np.asarray(rpy, dtype=float)
    cr, cp, cy = np.cos([r,p,y]); sr, sp, sy = np.sin([r,p,y])
    return np.array([[cy*cp, cy*sp*sr-sy*cr, cy*sp*cr+sy*sr],
                     [sy*cp, sy*sp*sr+cy*cr, sy*sp*cr-cy*sr],[-sp,cp*sr,cp*cr]])


def square_angle_difference(first, second):
    return float((first-second+np.pi/4) % (np.pi/2)-np.pi/4)


def die_world_angle(calibration, box, workspace):
    calibration.check_board(workspace)
    box = np.asarray(box, dtype=np.float32)
    if box.shape != (4,2) or not np.isfinite(box).all() or not cv2.isContourConvex(box.reshape(4,1,2)):
        raise ValueError('Need a finite convex die outline to calculate rotation.')
    registration = cv2.getPerspectiveTransform(workspace.corners.astype(np.float32), calibration.board)
    points = cv2.perspectiveTransform(box.reshape(1,4,2), calibration.matrix @ registration)[0]
    edge = points[1]-points[0]
    if not np.isfinite(points).all() or np.linalg.norm(edge) < .001:
        raise ValueError('Die orientation is degenerate.')
    return float(np.arctan2(edge[1],edge[0]))


def make_reference(calibration, box, workspace, actual_pose):
    actual_pose = np.asarray(actual_pose, dtype=float)
    if actual_pose.shape != (6,) or not np.isfinite(actual_pose).all():
        raise ValueError('Invalid reference robot pose.')
    return dict(version=1, edge_angle=die_world_angle(calibration, box, workspace),
                actual_pose=actual_pose.tolist(), matrix=calibration.matrix.tolist(),
                pickup_reference=calibration.pickup_reference.tolist(), tool=calibration.tool)


def validate_reference(calibration, reference):
    if reference.get('version') != 1 or reference.get('tool') != calibration.tool:
        raise ValueError('Gripper orientation reference is missing or incompatible. Teach it with Y.')
    for key, expected in [('matrix',calibration.matrix), ('pickup_reference',calibration.pickup_reference)]:
        actual = np.asarray(reference.get(key),dtype=float)
        if actual.shape != expected.shape or not np.isfinite(actual).all() or not np.allclose(actual,expected,atol=1e-8,rtol=1e-7):
            raise ValueError('Calibration changed since jaw alignment teaching. Press Y to re-teach.')
    pose = np.asarray(reference.get('actual_pose'),dtype=float)
    if pose.shape != (6,) or not np.isfinite(pose).all() or not np.isfinite(reference.get('edge_angle',float('nan'))):
        raise ValueError('Invalid jaw orientation reference.')


def apply_die_yaw(calibration, target, box, workspace):
    reference = getattr(calibration,'yaw_reference',None)
    if reference is None:
        raise ValueError('Auto rotation needs a jaw reference. Select an aligned reference die and press Y.')
    validate_reference(calibration,reference)
    desired = die_world_angle(calibration,box,workspace)
    # Infer the jaw axis from explicit physical alignment, rather than assume
    # that tool Y or Z happens to match the fitted gripper's fingers.
    a = reference['edge_angle']
    tool_axis = rotation(np.asarray(reference['actual_pose'])[3:]).T @ np.array([np.cos(a),np.sin(a),0.])
    world_axis = rotation(target[3:]) @ tool_axis
    if np.linalg.norm(world_axis[:2]) < .5:
        raise ValueError('Reference jaw axis is incompatible with the pickup orientation.')
    baseline = np.arctan2(world_axis[1],world_axis[0])
    delta = square_angle_difference(desired,baseline)
    result = target.copy()
    result[5] = (result[5]+delta+np.pi) % (2*np.pi)-np.pi
    return result


def load_reference(calibration):
    if not REFERENCE_FILE.exists():
        return None
    reference = json.loads(REFERENCE_FILE.read_text(encoding='utf-8'))
    validate_reference(calibration,reference)
    return reference


def save_reference(reference):
    temporary = REFERENCE_FILE.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(reference,indent=2),encoding='utf-8')
    temporary.replace(REFERENCE_FILE)
