import cv2
import math
import json
import numpy as np
from pathlib import Path
from config.settings import CONFIG

# 加载校准数据
CALIB_FILE = CONFIG['calibration']['file']
if Path(CALIB_FILE).exists():
    with open(CALIB_FILE, 'r') as f:
        calib = json.load(f)
else:
    calib = {}

LIPS_UP = [13]
LIPS_DOWN = [14]
LEFT_LIP_CORNER = 61
RIGHT_LIP_CORNER = 291
LEFT_EYE_UP = [159]
LEFT_EYE_DOWN = [145]
RIGHT_EYE_UP = [386]
RIGHT_EYE_DOWN = [374]
LEFT_PUPIL = 468
RIGHT_PUPIL = 473
LEFT_EYE = [33, 133]  # Outer and inner corners of left eye
RIGHT_EYE = [263, 362] # Outer and inner corners of right eye
MOUTH_LEFT = 61         # Left corner of mouth
MOUTH_RIGHT = 291       # Right corner of mouth
BROW_CENTER_LEFT = 107  # Left Eyebrow Center
BROW_CENTER_RIGHT = 336 # Right eyebrow center
# Eyebrow key points
LEFT_BROW_IDS = [65, 55, 52, 53, 46]
RIGHT_BROW_IDS = [295, 285, 282, 283, 276]
# Head orientation keypoints
NOSE_TIP = 1
MOUTH_LOWER_CENTER = 164  # 原错误设置为17
CHIN = 152
LEFT_EYE_INNER = 133
LEFT_EYE_OUTER = 33
RIGHT_EYE_INNER = 362
RIGHT_EYE_OUTER = 263

def calculate_features(lm, frame_shape):
    h, w = frame_shape[:2]
    features = {}
    h, w = frame_shape[:2]
    
    # Mouth width calculation
    ref = math.hypot(lm[LEFT_EYE_OUTER].x - lm[RIGHT_EYE_OUTER].x,
                     lm[LEFT_EYE_OUTER].y - lm[RIGHT_EYE_OUTER].y)
    raw_mw = math.hypot(lm[LEFT_LIP_CORNER].x - lm[RIGHT_LIP_CORNER].x,
                        lm[LEFT_LIP_CORNER].y - lm[RIGHT_LIP_CORNER].y) / ref
    base_mw = calib.get('mouth_width', raw_mw)
    features['mouth_width'] = raw_mw - base_mw

    # Mouth opening and closing
    mouth_open = max((lm[LIPS_DOWN[0]].y - lm[LIPS_UP[0]].y) * 5, 0)
    features['mouth_open'] = mouth_open

    # Eyes opening and closing
    for side, up_ids, down_ids in (('left', LEFT_EYE_UP, LEFT_EYE_DOWN),
                                   ('right', RIGHT_EYE_UP, RIGHT_EYE_DOWN)):
        raw = (sum(lm[i].y for i in down_ids)/len(down_ids) - sum(lm[i].y for i in up_ids)/len(up_ids)) * 10
        features[f'{side}_eyelid'] = min(max(raw, 0), 1)

    # Pupil position (based on eye keypoints)
    for side in ('left', 'right'):
        if side == 'left':
            inner = lm[LEFT_EYE_INNER]
            outer = lm[LEFT_EYE_OUTER]
            up = lm[LEFT_EYE_UP[0]]
            down = lm[LEFT_EYE_DOWN[0]]
            pupil = lm[LEFT_PUPIL]
        else:
            inner = lm[RIGHT_EYE_INNER]
            outer = lm[RIGHT_EYE_OUTER]
            up = lm[RIGHT_EYE_UP[0]]
            down = lm[RIGHT_EYE_DOWN[0]]
            pupil = lm[RIGHT_PUPIL]
        
        eye_width = outer.x - inner.x
        eye_height = down.y - up.y
        
        # Preventing division by zero errors
        if abs(eye_width) < 1e-4: eye_width = 1e-4
        if abs(eye_height) < 1e-4: eye_height = 1e-4
        
        # Normalized pupil position (center is 0 point)
        features[f'{side}_pupil_x'] = ((pupil.x - inner.x) / eye_width - 0.5) * 0.1
        features[f'{side}_pupil_y'] = ((pupil.y - up.y) / eye_height - 0.5) * 0.1

    # 3D head rotation (using solvePnP)//who can handdle this sh**t?
    image_points = np.array([
        (lm[NOSE_TIP].x * w, lm[NOSE_TIP].y * h),
        (lm[CHIN].x * w, lm[CHIN].y * h),
        (lm[LEFT_EYE_OUTER].x * w, lm[LEFT_EYE_OUTER].y * h),
        (lm[RIGHT_EYE_OUTER].x * w, lm[RIGHT_EYE_OUTER].y * h),
        (lm[MOUTH_LEFT].x * w, lm[MOUTH_LEFT].y * h),
        (lm[MOUTH_RIGHT].x * w, lm[MOUTH_RIGHT].y * h),
        (lm[BROW_CENTER_LEFT].x * w, lm[BROW_CENTER_LEFT].y * h),
        (lm[BROW_CENTER_RIGHT].x * w, lm[BROW_CENTER_RIGHT].y * h),
    ], dtype=np.float64)

    camera_matrix = np.array([
        [w * 0.9, 0, w/2],
        [0, w * 0.9, h/2],
        [0, 0, 1]
    ], dtype=np.float64)

    _, rvec, tvec = cv2.solvePnP(
        MODEL_POINTS,
        image_points,
        camera_matrix,
        np.zeros((4, 1), dtype=np.float64),
        flags=cv2.SOLVEPNP_EPNP
    )

    # Convert to Euler angles
    rmat = cv2.Rodrigues(rvec)[0]
    euler = rotation_matrix_to_euler(rmat)

    # Head calibration logic
    if not calib.get('head_calibrated'):
        # Initialize calibration (at this point the euler has been calculated)
        calib['head_pitch_offset'] = euler[0]
        calib['head_yaw_offset'] = euler[1]
        calib['head_roll_offset'] = euler[2]
        calib['head_calibrated'] = True
        # Save calibration data to file
        with open(CALIB_FILE, 'w') as f:
            json.dump(calib, f)
    
    # Apply Calibration Offset
    features['head_pitch'] = euler[0] - calib['head_pitch_offset']
    features['head_yaw'] = euler[1] - calib['head_yaw_offset']
    features['head_roll'] = euler[2] - calib['head_roll_offset']

    # Eyebrow lift
    for side, brow_ids in (('left', LEFT_BROW_IDS), ('right', RIGHT_BROW_IDS)):
        brow_y = sum(lm[i].y for i in brow_ids)/len(brow_ids)
        raw_brow = (lm[NOSE_TIP].y - brow_y) * 10
        base_b = calib.get(f'brow_{side}', raw_brow)
        features[f'{side}_brow'] = raw_brow - base_b

    # tooth opening and closing
    nose_tip = lm[NOSE_TIP]
    chin = lm[CHIN]
    lower_lip = lm[MOUTH_LOWER_CENTER]
    
    # Calculate reference distance (using reference points from configuration)
    ref_points = [lm[i] for i in CONFIG['calibration']['ref_points']]
    ref_distance = math.hypot(ref_points[1].x - ref_points[0].x,
                            ref_points[1].y - ref_points[0].y)
    
    # Calculate vertical distance
    vertical_dist = abs(chin.y - nose_tip.y)
    lower_lip_dist = abs(lower_lip.y - nose_tip.y)
    
    # Calculate the normalized opening and closing
    raw_teeth = max((vertical_dist - lower_lip_dist) / ref_distance * 5, 0)
    base_teeth = calib.get('teeth_open', raw_teeth)
    features['teeth_open'] = max(raw_teeth - base_teeth, 0)
    
    return features

def draw_preview(img, feats):
    y = 30
    for k, v in feats.items():
        cv2.putText(img, f"{k}: {v:.2f}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
        y += 25
    # Pupil arrow
    h, w = img.shape[:2]
    for side in ('left', 'right'):
        bx = int(w*(0.3 if side=='left' else 0.7)); by = h//2
        dx = int(feats[f'{side}_pupil_x']*100); dy = int(feats[f'{side}_pupil_y']*100)
        cv2.arrowedLine(img, (bx, by), (bx+dx, by+dy), (0,255,255), 2)
    # Head Arrow
    cv2.putText(img, f"Head Euler: ({feats['head_pitch']:.1f}, {feats['head_yaw']:.1f}, {feats['head_roll']:.1f})",
                (w-350, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
    return img

def rotation_matrix_to_euler(rmat):
    # 确保旋转矩阵是3x3
    assert rmat.shape == (3,3), "Invalid rotation matrix shape"
    
    # Extract Euler angles (ZYX order)
    sy = math.sqrt(rmat[0,0]**2 + rmat[1,0]**2)
    singular = sy < 1e-6

    if not singular:
        x = math.atan2(rmat[2,1], rmat[2,2])  # Pitch
        y = math.atan2(-rmat[2,0], sy)        # Yaw
        z = math.atan2(rmat[1,0], rmat[0,0])  # Roll
    else:
        x = math.atan2(-rmat[1,2], rmat[1,1])
        y = math.atan2(-rmat[2,0], sy)
        z = 0

    return np.degrees([x, y, z])

MODEL_POINTS = np.array([
    (0.0, 0.0, 0.0),         # NOSE_TIP
    (0.0, -330.0, -65.0),    # CHIN
    (-165.0, 170.0, -135.0), # LEFT_EYE_OUTER
    (165.0, 170.0, -135.0),  # RIGHT_EYE_OUTER
    (-75.0, -40.0, -125.0),  # MOUTH_LEFT
    (75.0, -40.0, -125.0),   # MOUTH_RIGHT
    (-60.0, 130.0, -110.0),  # BROW_CENTER_LEFT
    (60.0, 130.0, -110.0)    # BROW_CENTER_RIGHT
], dtype=np.float64)