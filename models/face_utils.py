from face_constants import *
import mediapipe as mp
mp_face_mesh = mp.solutions.face_mesh
FACE_CONNECTIONS = mp_face_mesh.FACEMESH_TESSELATION

class HeadRotationCalculator:
    def __init__(self, config):
        self.config = config
        self.calib_points = config['head_calibration']['calib_points']
        try:
            with open(config['head_calibration']['file'], 'r') as f:
                self.head_calib = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            self.head_calib = {'pitch':0, 'yaw':0, 'roll':0}

    def calculate_head_rotation(self, lm, frame_shape, features):
        try:
            h, w = frame_shape[:2]
            image_points = self._get_image_points(lm, w, h)
            if image_points is None: return features
            
            euler = self._solve_head_pose(image_points, w, h)
            features['head_pitch'] = -(euler[0] - self.head_calib['pitch'])
            features['head_yaw'] = -(euler[1] - self.head_calib['yaw'])
            features['head_roll'] = euler[2] - self.head_calib['roll']
        except Exception as e:
            print(f"Head rotation error: {str(e)}")
            features.update({'head_pitch':0, 'head_yaw':0, 'head_roll':0})
        return features

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

def calculate_features(lm, frame_shape):
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

def save_head_calibration(features):
    calib_data = {
        'pitch': features['head_pitch'],
        'yaw': features['head_yaw'],
        'roll': features['head_roll']
    }
    with open(CONFIG['head_calibration']['file'], 'w') as f:
        json.dump(calib_data, f)
    print(f"Head calibration saved to {CONFIG['head_calibration']['file']}")

def draw_preview(img, feats, lm):  # 添加 lm 参数
    y = 30
    for k, v in feats.items():
        cv2.putText(img, f"{k}: {v:.2f}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
        y += 25
    
    # 绘制面部网格
    h, w = img.shape[:2]
    for connection in FACE_CONNECTIONS:
        start_idx, end_idx = connection
        if start_idx < len(lm) and end_idx < len(lm):
            x1 = int(lm[start_idx].x * w)
            y1 = int(lm[start_idx].y * h)
            x2 = int(lm[end_idx].x * w)
            y2 = int(lm[end_idx].y * h)
            cv2.line(img, (x1, y1), (x2, y2), (100, 100, 100), 1)
    
    # 定义左右眼和眉毛的关键点索引集合
    LEFT_POINTS = set(LEFT_PUPIL_IDS + LEFT_EYE_UP + LEFT_EYE_DOWN + LEFT_BROW_IDS)
    RIGHT_POINTS = set(RIGHT_PUPIL_IDS + RIGHT_EYE_UP + RIGHT_EYE_DOWN + RIGHT_BROW_IDS)

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