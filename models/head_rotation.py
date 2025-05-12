from face_constants import *

class HeadRotationCalculator:
    def __init__(self, config):
        self.calib_file = config['calibration']['file']
        self.calib = self._load_calibration()
        self.ref_points = config['calibration']['ref_points']

    def _load_calibration(self):
        try:
            with open(self.calib_file, 'r') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {}

    def _save_calibration(self):
        with open(self.calib_file, 'w') as f:
            json.dump(self.calib, f)

    def calculate_head_rotation(self, lm, frame_shape, features):
        # Adding null protection
        features.setdefault('head_pitch', 0.0)
        features.setdefault('head_yaw', 0.0)
        features.setdefault('head_roll', 0.0)
    
        try:
            h, w = frame_shape[:2]
            image_points = self._get_image_points(lm, w, h)
            euler = self._solve_head_pose(image_points, w, h)
        
            if not self.calib.get('head_calibrated'):
                self._initialize_calibration(euler)
            
            self._apply_calibration(euler, features)
        
        except Exception as e:
            print(f"Head rotation calculation failed: {str(e)}")
            # 保留上次有效值（如果有）
            features['head_pitch'] = features.get('head_pitch', 0.0)
            features['head_yaw'] = features.get('head_yaw', 0.0)
            features['head_roll'] = features.get('head_roll', 0.0)
    
        return features

    def _get_image_points(self, lm, w, h):
        required_indices = [
            NOSE_TIP, CHIN, LEFT_EYE_OUTER, RIGHT_EYE_OUTER,
            MOUTH_LEFT, MOUTH_RIGHT, BROW_CENTER_LEFT, BROW_CENTER_RIGHT
        ]
    
        # 验证所有必要地标点存在
        for idx in required_indices:
            if not hasattr(lm[idx], 'x') or not hasattr(lm[idx], 'y'):
                raise ValueError(f"Key point {idx} does not exist or is malformed")
    
        return np.array([
            (lm[NOSE_TIP].x * w, lm[NOSE_TIP].y * h),
            (lm[CHIN].x * w, lm[CHIN].y * h),
            (lm[LEFT_EYE_OUTER].x * w, lm[LEFT_EYE_OUTER].y * h),
            (lm[RIGHT_EYE_OUTER].x * w, lm[RIGHT_EYE_OUTER].y * h),
            (lm[MOUTH_LEFT].x * w, lm[MOUTH_LEFT].y * h),
            (lm[MOUTH_RIGHT].x * w, lm[MOUTH_RIGHT].y * h),
            (lm[BROW_CENTER_LEFT].x * w, lm[BROW_CENTER_LEFT].y * h),
            (lm[BROW_CENTER_RIGHT].x * w, lm[BROW_CENTER_RIGHT].y * h)
        ], dtype=np.float64)

    def _solve_head_pose(self, image_points, w, h):
        # Adding type checking
        if not isinstance(MODEL_POINTS, np.ndarray):
            raise TypeError(f"MODEL_POINTS should be a NumPy array, the actual type is {type(MODEL_POINTS)}")
    
        if not isinstance(image_points, np.ndarray):
            raise TypeError(f"image_points should be a NumPy array, the actual type is {type(image_points)}")

        camera_matrix = np.array([
            [w * 0.9, 0, w/2],
            [0, w * 0.9, h/2],
            [0, 0, 1]
        ], dtype=np.float64)

        # Adding null protection
        if MODEL_POINTS.size == 0 or image_points.size == 0:
            return np.zeros(3)

        try:
            _, rvec, tvec = cv2.solvePnP(
                MODEL_POINTS,
                image_points,
                camera_matrix,
                np.zeros((4, 1), dtype=np.float64),
                flags=cv2.SOLVEPNP_EPNP
            )
        except cv2.error as e:
            print(f"SolvePnP error: {str(e)}")
            print(f"MODEL_POINTS shape: {MODEL_POINTS.shape}")
            print(f"image_points shape: {image_points.shape}")
            return np.zeros(3)

        rmat = cv2.Rodrigues(rvec)[0]
        return self._rotation_matrix_to_euler(rmat)

    def _rotation_matrix_to_euler(self, rmat):
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

    def _initialize_calibration(self, euler):
        self.calib.update({
            'head_pitch_offset': euler[0],
            'head_yaw_offset': euler[1],
            'head_roll_offset': euler[2],
            'head_calibrated': True
        })
        self._save_calibration()

    def _apply_calibration(self, euler, features):
        features['head_pitch'] = euler[0] - self.calib.get('head_pitch_offset', 0)
        features['head_yaw'] = euler[1] - self.calib.get('head_yaw_offset', 0)
        features['head_roll'] = euler[2] - self.calib.get('head_roll_offset', 0)