"""Learn a repeatable world offset + rotating jaw-centre offset from manual previews.

No robot or camera commands. This is feed-forward compensation, not visual servoing.
A leave-one-out check gates use; application stays inside sampled positions/angles.
"""
import hashlib
import json
from pathlib import Path
import cv2
import numpy as np
from dice_camera_calibration import pose_array

SAMPLES_FILE = Path(__file__).with_name('dice_alignment_samples.json')


def binding(cal):
    ref = getattr(cal, 'yaw_reference', None)
    if ref is None:
        raise ValueError('Teach jaw reference with Y before alignment learning.')
    payload = dict(matrix=cal.matrix.tolist(), board=cal.board.tolist(),
                   reference=cal.pickup_reference.tolist(), tool=cal.tool,
                   intrinsics=[a.tolist() for a in cal.intrinsics],
                   grip_z=float(np.mean([p[2] for p in cal.poses])), jaw=ref)
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def angle_delta(a, b):
    return float((a-b+np.pi) % (2*np.pi)-np.pi)


def check_pair(planned, corrected):
    a, b = pose_array(planned), pose_array(corrected)
    if not np.allclose(a[2:5], b[2:5], atol=1e-9, rtol=0):
        raise ValueError('Learning must keep Z, roll and pitch fixed.')
    if np.linalg.norm(b[:2]-a[:2]) > .020 or abs(angle_delta(b[5],a[5])) > np.deg2rad(15)+1e-9:
        raise ValueError('Correction exceeds 20 mm or 15 degrees; inspect mapping/jaw reference first.')
    return a, b


def design(yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[1.,0.,c,-s],[0.,1.,s,c]])


def solve(poses, corrected):
    matrix = np.vstack([design(p[5]) for p in poses])
    if np.linalg.matrix_rank(matrix) != 4 or np.linalg.cond(matrix) > 100:
        raise ValueError('Need more varied cube rotations (spread at least 30 degrees).')
    offset = np.linalg.lstsq(matrix,(corrected[:,:2]-poses[:,:2]).ravel(),rcond=None)[0]
    turns = np.array([angle_delta(b[5],a[5]) for a,b in zip(poses,corrected)])
    return offset, float(np.mean(turns))


def fit_samples(data):
    samples = data.get('samples', [])
    if len(samples) < 8:
        raise ValueError(f'Need at least 8 aligned examples; saved {len(samples)}. Spread positions and rotations.')
    pairs = [check_pair(s['planned'],s['corrected']) for s in samples]
    poses, corrected = np.array([p[0] for p in pairs]), np.array([p[1] for p in pairs])
    if np.ptp(poses[:,2]) > 1e-6:
        raise ValueError('Inspection height changed; use one sample set at the same height.')
    angles = np.array([angle_delta(p[5],poses[0,5]) for p in poses])
    hull = cv2.convexHull(poses[:,:2].astype(np.float32))
    if np.any(np.ptp(poses[:,:2],axis=0) < .03) or cv2.contourArea(hull) < .0009:
        raise ValueError('Spread examples across the board in both X and Y, not one cube location.')
    if np.ptp(angles) < np.deg2rad(30):
        raise ValueError('Need cube rotation spread of at least 30 degrees.')
    errors, turns = [], []
    for index in range(len(poses)):
        keep = np.arange(len(poses)) != index
        offset, turn = solve(poses[keep],corrected[keep])
        errors.append(np.linalg.norm(design(poses[index,5]) @ offset - (corrected[index,:2]-poses[index,:2])))
        turns.append(abs(angle_delta(corrected[index,5],poses[index,5])-turn))
    if max(errors) > .002 or max(turns) > np.deg2rad(2):
        raise ValueError(f'Independent leave-one-out errors: {max(errors)*1000:.2f} mm, '
                         f'{np.degrees(max(turns)):.2f} deg; limits 2 mm / 2 deg. '
                         'Examples disagree; do not enable automatic correction.')
    offset, turn = solve(poses,corrected)
    return dict(offset=offset, turn=turn, hull=hull, angle_origin=poses[0,5],
                angle_min=min(angles), angle_max=max(angles), z=poses[0,2],
                max_error_mm=max(errors)*1000, max_error_deg=np.degrees(max(turns)),
                binding=data['binding'], sample_count=len(samples))


def load_samples(cal, path=None):
    path = SAMPLES_FILE if path is None else Path(path)
    expected = binding(cal)
    if not path.exists():
        return dict(version=1,binding=expected,samples=[])
    data = json.loads(path.read_text(encoding='utf-8'))
    if data.get('version') != 1 or data.get('binding') != expected:
        raise ValueError('Alignment samples belong to a different mapping/tool/jaw reference. Keep them separate and re-teach.')
    return data


def append_sample(cal, report, path=None):
    path = SAMPLES_FILE if path is None else Path(path)
    data = load_samples(cal,path)
    a,b = check_pair(report['planned'],report['corrected_command'])
    for sample in data['samples']:
        old = pose_array(sample['planned'])
        if abs(old[2]-a[2]) > 1e-6:
            raise ValueError('Inspection height differs from saved examples; keep 30 mm clearance.')
        if np.linalg.norm(old[:2]-a[:2]) < .01 and abs(angle_delta(old[5],a[5])) < np.deg2rad(10):
            raise ValueError('Example too similar to an existing one. Move or rotate the cube before next example.')
    from datetime import datetime, timezone
    data['samples'].append(dict(planned=a.tolist(),corrected=b.tolist(),
                                actual=report['corrected_actual'], recorded_at=datetime.now(timezone.utc).isoformat()))
    temporary=path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(data,indent=2),encoding='utf-8')
    temporary.replace(path)
    print(f'Alignment example {len(data["samples"])} saved. Original pickup calibration unchanged.')
    try:
        model = fit_samples(data)
    except ValueError as error:
        print(f'Automatic correction not ready: {error}')
        return
    print(f'ALIGNMENT READY: {model["sample_count"]} examples; leave-one-out maximum '
          f'{model["max_error_mm"]:.2f} mm / {model["max_error_deg"]:.2f} deg. '
          'Restart with --use-alignment for a preview on NEW positions; no automatic gripping.')


def load_model(cal):
    model=fit_samples(load_samples(cal))
    if abs(model['z']-(np.mean([p[2] for p in cal.poses])+.030)) > 1e-6:
        raise ValueError('Alignment was not taught at 30 mm preview clearance.')
    return model


def apply_alignment(cal, target):
    model=cal.alignment_model
    if model['binding'] != binding(cal):
        raise ValueError('Alignment binding changed. Re-teach before applying.')
    a=pose_array(target)
    if cv2.pointPolygonTest(model['hull'],tuple(map(float,a[:2])),False) < 0:
        raise ValueError('Outside learned alignment positions. Teach examples around this position first.')
    yaw=angle_delta(a[5],model['angle_origin'])
    if not model['angle_min']-1e-8 <= yaw <= model['angle_max']+1e-8:
        raise ValueError('Outside learned rotation range. Teach examples at this angle first.')
    b=a.copy()
    b[:2]+=design(a[5]) @ model['offset']
    b[5]=a[5]+model['turn']
    check_pair(a,b)
    # Corrected command must also stay on the physically taught board.
    board=cv2.perspectiveTransform(cal.board.reshape(1,4,2),cal.matrix)[0]
    if cv2.pointPolygonTest(board.astype(np.float32),tuple(map(float,b[:2])),False) < 0:
        raise ValueError('Corrected tool position would leave the board.')
    return b
