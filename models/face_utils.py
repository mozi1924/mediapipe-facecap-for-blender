from face_constants import *

class HeadRotationCalculator:
    def __init__(self, config):
        self.config = config
        self.calib_points = config['head_calibration']['calib_points']
        try:
            with open(config['head_calibration']['file'], 'r') as f:
                self.head_calib = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            self.head_calib = {'pitch':0, 'yaw':0, 'roll':0}
            
    def calculate_head_rotation(self, lm, frame_shape, features, raw_features):  # 新增 raw_features 参数
        try:
            h, w = frame_shape[:2]
            image_points = self._get_image_points(lm, w, h)
            if image_points is None: return features, raw_features
            
            euler = self._solve_head_pose(image_points, w, h)
            raw_head_pitch = -euler[0]
            raw_head_yaw = -euler[1]
            raw_head_roll = euler[2]
            
            # 校准后的特征值
            features['head_pitch'] = raw_head_pitch - self.head_calib['pitch']
            features['head_yaw'] = raw_head_yaw - self.head_calib['yaw']
            features['head_roll'] = raw_head_roll - self.head_calib['roll']
            
            # 将原始值添加到 raw_features
            raw_features['_raw_head_pitch'] = raw_head_pitch
            raw_features['_raw_head_yaw'] = raw_head_yaw
            raw_features['_raw_head_roll'] = raw_head_roll
            
        except Exception as e:
            print(f"Head rotation error: {str(e)}")
            features.update({'head_pitch':0, 'head_yaw':0, 'head_roll':0})
        return features, raw_features  # 返回更新后的 features 和 raw_features

    def _get_image_points(self, lm, w, h):
        image_points = []
        for idx in self.calib_points:
            if idx >= len(lm) or not hasattr(lm[idx], 'x'):
                return None
            image_points.append([lm[idx].x * w, lm[idx].y * h])
        return np.array(image_points, dtype=np.float64)

    def _solve_head_pose(self, image_points, w, h):
        if MODEL_POINTS.shape[0] != image_points.shape[0]:
            raise ValueError("3D/2D points mismatch")
        
        camera_matrix = np.array([
            [w, 0, w/2],
            [0, w, h/2],
            [0, 0, 1]
        ], dtype=np.float64)
        
        _, rvec, _ = cv2.solvePnP(
            MODEL_POINTS,
            image_points,
            camera_matrix,
            np.zeros((4, 1), dtype=np.float64),
            flags=cv2.SOLVEPNP_ITERATIVE
        )
        return self._rotation_matrix_to_euler(cv2.Rodrigues(rvec)[0])

    def _rotation_matrix_to_euler(self, R):
        x = math.atan2(R[2,1], R[2,2])
        y = math.atan2(-R[2,0], math.hypot(R[0,0], R[1,0]))
        z = math.atan2(R[1,0], R[0,0])
        return np.degrees([x, y, z])

# 初始化模块级组件
head_rotator = HeadRotationCalculator(CONFIG)

def calculate_mouth_features(lm, features, raw_features):
    # 计算嘴巴宽度
    ref = math.hypot(lm[LEFT_EYE_OUTER].x - lm[RIGHT_EYE_OUTER].x,
                     lm[LEFT_EYE_OUTER].y - lm[RIGHT_EYE_OUTER].y)
    raw_mw = math.hypot(lm[LEFT_LIP_CORNER].x - lm[RIGHT_LIP_CORNER].x,
                        lm[LEFT_LIP_CORNER].y - lm[RIGHT_LIP_CORNER].y) / ref
    base_mw = calib.get('mouth_width', raw_mw)
    features['mouth_width'] = raw_mw - base_mw
    raw_features['_raw_mouth_width'] = raw_mw  # 原始值存到raw_features

    # 计算嘴巴开合
    mouth_open = max((lm[LIPS_DOWN[0]].y - lm[LIPS_UP[0]].y) * 5, 0)
    features['mouth_open'] = mouth_open
    
    return features, raw_features

def calculate_eye_features(lm, features):
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
            pupil_ids = LEFT_PUPIL_IDS  # 使用环形关键点
        else:
            inner = lm[RIGHT_EYE_INNER]
            outer = lm[RIGHT_EYE_OUTER]
            up = lm[RIGHT_EYE_UP[0]]
            down = lm[RIGHT_EYE_DOWN[0]]
            pupil_ids = RIGHT_PUPIL_IDS  # 使用环形关键点
        
        # 计算瞳孔中心（取环形关键点平均值）
        pupil_x_avg = sum(lm[i].x for i in pupil_ids) / len(pupil_ids)
        pupil_y_avg = sum(lm[i].y for i in pupil_ids) / len(pupil_ids)
        
        eye_width = outer.x - inner.x
        eye_height = down.y - up.y
        
        # 防止除零错误
        if abs(eye_width) < 1e-4: eye_width = 1e-4
        if abs(eye_height) < 1e-4: eye_height = 1e-4
        
        # 归一化瞳孔位置（中心为0点）
        features[f'{side}_pupil_x'] = ((pupil_x_avg - inner.x) / eye_width - 0.5) * 0.1
        features[f'{side}_pupil_y'] = ((pupil_y_avg - up.y) / eye_height - 0.5) * 0.1
        
    return features

def calculate_eyebrow_features(lm, features, raw_features):
    # 计算眉毛高度
    for side, brow_ids in (('left', LEFT_BROW_IDS), ('right', RIGHT_BROW_IDS)):
        brow_y = sum(lm[i].y for i in brow_ids)/len(brow_ids)
        raw_brow = (lm[NOSE_TIP].y - brow_y) * 10
        base_b = calib.get(f'brow_{side}', raw_brow)
        features[f'{side}_brow'] = raw_brow - base_b
        raw_features[f'_raw_{side}_brow'] = raw_brow
        
    return features, raw_features

def calculate_teeth_features(lm, features, raw_features):
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
    base_teeth = calib.get('teeth_open', raw_teeth)
    features['teeth_open'] = max(raw_teeth - base_teeth, 0)
    raw_features['_raw_teeth_open'] = raw_teeth
    
    return features, raw_features

def calculate_features(lm, frame_shape):
    features, raw_features = {}, {}
    
    # 分步骤计算各类特征
    features, raw_features = calculate_mouth_features(lm, features, raw_features)
    features = calculate_eye_features(lm, features)
    features, raw_features = calculate_eyebrow_features(lm, features, raw_features)
    features, raw_features = calculate_teeth_features(lm, features, raw_features)
    
    # 计算头部旋转并传递 raw_features
    features, raw_features = head_rotator.calculate_head_rotation(lm, frame_shape, features, raw_features)  # 修改此行
    
    # 返回前清理 features
    features_clean = {k: v for k, v in features.items() if not k.startswith('_raw_')}
    return features_clean, raw_features

def save_head_calibration(raw_features):
    calib_data = {
        'pitch': raw_features.get('_raw_head_pitch', 0),
        'yaw': raw_features.get('_raw_head_yaw', 0),
        'roll': raw_features.get('_raw_head_roll', 0)
    }
    with open(CONFIG['head_calibration']['file'], 'w') as f:
        json.dump(calib_data, f)
    print(f"Head calibration saved to {CONFIG['head_calibration']['file']}")

def draw_preview(img, feats, lm):  # 添加 lm 参数
    y = 30
    for k, v in feats.items():
        cv2.putText(img, f"{k}: {v:.2f}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
        y += 25
    
    # 定义左右眼和眉毛的关键点索引集合
    LEFT_POINTS = set(LEFT_PUPIL_IDS + LEFT_EYE_UP + LEFT_EYE_DOWN)
    RIGHT_POINTS = set(RIGHT_PUPIL_IDS + RIGHT_EYE_UP + RIGHT_EYE_DOWN)

    # 绘制关键点（可选）
    h, w = img.shape[:2]
    for idx, point in enumerate(lm):
        x = int(point.x * w)
        y = int(point.y * h)
        if idx in LEFT_POINTS:
            color = (0, 0, 255)  # 红色（BGR格式）
        elif idx in RIGHT_POINTS:
            color = (255, 0, 0)  # 蓝色（BGR格式）
        else:
            continue  # 其他关键点不绘制
        cv2.circle(img, (x, y), 2, color, -1)  # 增大点半径到2像素
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