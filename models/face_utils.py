from face_constants import *
import cv2
import numpy as np
import math
import json

class HeadRotationCalculator:
    def __init__(self):
        self.calib_points = CONFIG['head_calibration']['calib_points']
    
    def calculate_head_rotation(self, lm, frame_shape):
        features = {}
        raw_features = {}
        try:
            h, w = frame_shape[:2]
            image_points = np.array([
                [lm[idx].x * w, lm[idx].y * h] 
                for idx in self.calib_points
            ], dtype=np.float64)
            
            if len(image_points) == 0:
                return features, raw_features
            
            # 使用更精确的相机矩阵
            focal_length = w * 1.5
            camera_matrix = np.array([
                [focal_length, 0, w/2],
                [0, focal_length, h/2],
                [0, 0, 1]
            ], dtype=np.float64)
            
            _, rvec, _ = cv2.solvePnP(
                MODEL_POINTS[:len(self.calib_points)],
                image_points,
                camera_matrix,
                None,
                flags=cv2.SOLVEPNP_ITERATIVE
            )
            
            euler = self._rotation_matrix_to_euler(cv2.Rodrigues(rvec)[0])
            head_calib = get_head_calib()
            
            # 计算并存储结果
            raw_features.update({
                '_raw_head_pitch': -euler[0],
                '_raw_head_yaw': -euler[1],
                '_raw_head_roll': euler[2]
            })
            
            features.update({
                'head_pitch': raw_features['_raw_head_pitch'] - head_calib.get('pitch', 0),
                'head_yaw': raw_features['_raw_head_yaw'] - head_calib.get('yaw', 0),
                'head_roll': raw_features['_raw_head_roll'] - head_calib.get('roll', 0)
            })
            
        except Exception as e:
            print(f"Head rotation error: {e}")
            features.update({'head_pitch': 0, 'head_yaw': 0, 'head_roll': 0})
            
        return features, raw_features

    def _rotation_matrix_to_euler(self, R):
        x = math.atan2(R[2,1], R[2,2])
        y = math.atan2(-R[2,0], math.hypot(R[0,0], R[1,0]))
        z = math.atan2(R[1,0], R[0,0])
        return np.degrees([x, y, z])

# 模块级组件
head_rotator = HeadRotationCalculator()

def calculate_mouth_features(lm):
    features, raw_features = {}, {}
    
    # 眼睛外角作为参考距离
    ref_dist = math.hypot(
        lm[LEFT_EYE_OUTER].x - lm[RIGHT_EYE_OUTER].x,
        lm[LEFT_EYE_OUTER].y - lm[RIGHT_EYE_OUTER].y
    )
    
    # 嘴巴宽度
    raw_mw = math.hypot(
        lm[LEFT_LIP_CORNER].x - lm[RIGHT_LIP_CORNER].x,
        lm[LEFT_LIP_CORNER].y - lm[RIGHT_LIP_CORNER].y
    ) / ref_dist
    
    features['mouth_width'] = raw_mw - get_calib().get('mouth_width', raw_mw)
    raw_features['_raw_mouth_width'] = raw_mw
    
    # 嘴巴开合
    features['mouth_open'] = max((lm[LIPS_DOWN[0]].y - lm[LIPS_UP[0]].y) * 5, 0)
    
    return features, raw_features

def calculate_eye_features(lm):
    features, raw_features = {}, {}
    # 眼睛开合
    for side, up_ids, down_ids in (('left', LEFT_EYE_UP, LEFT_EYE_DOWN),
                                  ('right', RIGHT_EYE_UP, RIGHT_EYE_DOWN)):
        raw = (sum(lm[i].y for i in down_ids)/len(down_ids) - sum(lm[i].y for i in up_ids)/len(up_ids)) * 10
        features[f'{side}_eyelid'] = min(max(raw, 0), 1)

    # 瞳孔位置计算
    for side in ('left', 'right'):
        if side == 'left':
            inner = lm[LEFT_EYE_INNER]
            outer = lm[LEFT_EYE_OUTER]
            up = lm[LEFT_EYE_UP[0]]
            down = lm[LEFT_EYE_DOWN[0]]
            pupil_ids = LEFT_PUPIL_IDS
        else:
            inner = lm[RIGHT_EYE_INNER]
            outer = lm[RIGHT_EYE_OUTER]
            up = lm[RIGHT_EYE_UP[0]]
            down = lm[RIGHT_EYE_DOWN[0]]
            pupil_ids = RIGHT_PUPIL_IDS
        
        # 计算瞳孔中心
        pupil_x_avg = sum(lm[i].x for i in pupil_ids) / len(pupil_ids)
        pupil_y_avg = sum(lm[i].y for i in pupil_ids) / len(pupil_ids)
        
        eye_width = outer.x - inner.x
        eye_height = down.y - up.y
        
        # 防止除零错误
        if abs(eye_width) < 1e-4: eye_width = 1e-4
        if abs(eye_height) < 1e-4: eye_height = 1e-4
        
        # 归一化瞳孔位置
        features[f'{side}_pupil_x'] = ((pupil_x_avg - inner.x) / eye_width - 0.5) * 0.1
        features[f'{side}_pupil_y'] = ((pupil_y_avg - up.y) / eye_height - 0.5) * 0.1
        
    return features, raw_features

def calculate_eyebrow_features(lm):
    features, raw_features = {}, {}
    # 计算眉毛高度
    for side, brow_ids in (('left', LEFT_BROW_IDS), ('right', RIGHT_BROW_IDS)):
        brow_y = sum(lm[i].y for i in brow_ids)/len(brow_ids)
        raw_brow = (lm[NOSE_TIP].y - brow_y) * 10
        base_b = get_calib().get(f'brow_{side}', raw_brow)
        features[f'{side}_brow'] = raw_brow - base_b
        raw_features[f'_raw_{side}_brow'] = raw_brow
        
    return features, raw_features

def calculate_teeth_features(lm):
    features, raw_features = {}, {}
    # 计算牙齿开合
    nose_tip = lm[NOSE_TIP]
    chin = lm[CHIN]
    lower_lip = lm[MOUTH_LOWER_CENTER]
    
    # 计算参考距离
    ref_points = [lm[i] for i in CONFIG['calibration']['ref_points']]
    ref_distance = math.hypot(ref_points[1].x - ref_points[0].x,
                            ref_points[1].y - ref_points[0].y)
    
    # 计算垂直距离
    vertical_dist = abs(chin.y - nose_tip.y)
    lower_lip_dist = abs(lower_lip.y - nose_tip.y)
    
    # 计算归一化的牙齿开合度
    raw_teeth = max((vertical_dist - lower_lip_dist) / ref_distance * 5, 0)
    base_teeth = get_calib().get('teeth_open', raw_teeth)
    features['teeth_open'] = max(raw_teeth - base_teeth, 0)
    raw_features['_raw_teeth_open'] = raw_teeth
    
    return features, raw_features

def calculate_features(lm, frame_shape):
    features, raw_features = {}, {}
    
    # 并行计算不同特征
    results = [
        calculate_mouth_features(lm),
        calculate_eye_features(lm),
        calculate_eyebrow_features(lm),
        calculate_teeth_features(lm),
        head_rotator.calculate_head_rotation(lm, frame_shape)
    ]
    
    # 合并结果
    for result in results:
        features.update(result[0])
        raw_features.update(result[1])
    
    # 返回前清理 features
    features_clean = {k: v for k, v in features.items() if not k.startswith('_raw_')}
    return features_clean, raw_features

def save_calibration(raw_features):
    calib_data = {
        'mouth_width': raw_features.get('_raw_mouth_width', 0),
        'brow_left': raw_features.get('_raw_left_brow', 0),
        'brow_right': raw_features.get('_raw_right_brow', 0),
        'teeth_open': raw_features.get('_raw_teeth_open', 0)
    }
    with open(CONFIG['calibration']['file'], 'w') as f:
        json.dump(calib_data, f)
    print(f"Facial calibration saved")
    
    global _calib_cache
    _calib_cache = {"data": {}, "mtime": 0}
    print(f"Facial calibration saved and cache reset")

def save_head_calibration(raw_features):
    calib_data = {
        'pitch': raw_features.get('_raw_head_pitch', 0),
        'yaw': raw_features.get('_raw_head_yaw', 0),
        'roll': raw_features.get('_raw_head_roll', 0)
    }
    with open(CONFIG['head_calibration']['file'], 'w') as f:
        json.dump(calib_data, f)
    print(f"Head calibration saved")
    # 重置缓存以确保立即生效
    global _head_calib_cache
    _head_calib_cache = {"data": {}, "mtime": 0}
    print(f"Head calibration saved and cache reset")

def draw_preview(img, feats, lm):
    y = 30
    for k, v in feats.items():
        cv2.putText(img, f"{k}: {v:.2f}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
        y += 25
    
    # 定义左右眼和眉毛的关键点索引集合
    LEFT_POINTS = set(LEFT_PUPIL_IDS + LEFT_EYE_UP + LEFT_EYE_DOWN)
    RIGHT_POINTS = set(RIGHT_PUPIL_IDS + RIGHT_EYE_UP + RIGHT_EYE_DOWN)

    # 绘制关键点
    h, w = img.shape[:2]
    for idx, point in enumerate(lm):
        x = int(point.x * w)
        y = int(point.y * h)
        if idx in LEFT_POINTS:
            color = (0, 0, 255)  # 红色
        elif idx in RIGHT_POINTS:
            color = (255, 0, 0)  # 蓝色
        else:
            continue
        cv2.circle(img, (x, y), 2, color, -1)
    
    # 瞳孔位置指示
    h, w = img.shape[:2]
    for side in ('left', 'right'):
        bx = int(w*(0.3 if side=='left' else 0.7))
        by = h//2
        dx = int(feats.get(f'{side}_pupil_x', 0)*100)
        dy = int(feats.get(f'{side}_pupil_y', 0)*100)
        cv2.arrowedLine(img, (bx, by), (bx+dx, by+dy), (0,255,255), 2)
    
    # 头部姿态信息
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