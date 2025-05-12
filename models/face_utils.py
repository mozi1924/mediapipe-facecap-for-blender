from pathlib import Path
from face_constants import *
from .head_rotation import HeadRotationCalculator

# Loading the Head Rotation Calculator
head_rotator = HeadRotationCalculator(CONFIG)

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

    # Calculating head rotation
    features = head_rotator.calculate_head_rotation(lm, frame_shape, features)
    
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
    head_info = "Head Euler: "
    try:
        head_info += f"({feats.get('head_pitch', 0.0):.1f}, "
        head_info += f"{feats.get('head_yaw', 0.0):.1f}, "
        head_info += f"{feats.get('head_roll', 0.0):.1f})"
    except KeyError:
        head_info += "(N/A)"
    
    cv2.putText(img, head_info, (w-350, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
    return img