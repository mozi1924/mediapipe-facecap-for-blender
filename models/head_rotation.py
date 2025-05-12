from face_constants import *
import math

class HeadRotationCalculator:
    def __init__(self, config):
        # 移除校准文件相关初始化
        pass  # 完全不需要config参数，但为保持接口兼容性保留构造参数

    def calculate_head_rotation(self, lm, frame_shape, features):
        features.setdefault('head_pitch', 0.0)
        features.setdefault('head_yaw', 0.0)
        features.setdefault('head_roll', 0.0)
    
        try:
            h, w = frame_shape[:2]
            image_points = self._get_image_points(lm, w, h)
            
            if image_points is None:
                return features
                
            euler = self._solve_head_pose(image_points, w, h)
            
            # 直接赋值，不再应用校准
            features['head_pitch'] = -euler[0]  # 保持符号反转
            features['head_yaw'] = -euler[1]
            features['head_roll'] = euler[2]
        
        except Exception as e:
            print(f"Head rotation calculation failed: {str(e)}")
            features['head_pitch'] = features.get('head_pitch', 0.0)
            features['head_yaw'] = features.get('head_yaw', 0.0)
            features['head_roll'] = features.get('head_roll', 0.0)
    
        return features

    def _get_image_points(self, lm, w, h):
        # 保持不变
        required_indices = [1, 9, 57, 130, 287, 359]
        
        image_points = []
        for idx in required_indices:
            if not (0 <= idx < len(lm)) or not hasattr(lm[idx], 'x') or not hasattr(lm[idx], 'y'):
                return None
            image_points.append([lm[idx].x * w, lm[idx].y * h])
        
        return np.array(image_points, dtype=np.float64)

    def _solve_head_pose(self, image_points, w, h):
        # 保持不变
        if MODEL_POINTS.shape[0] != image_points.shape[0]:
            raise ValueError(f"3D/2D points mismatch: {MODEL_POINTS.shape[0]} vs {image_points.shape[0]}")

        focal_length = w
        camera_matrix = np.array([
            [focal_length, 0, w/2],
            [0, focal_length, h/2],
            [0, 0, 1]
        ], dtype=np.float64)

        _, rvec, _ = cv2.solvePnP(
            MODEL_POINTS,
            image_points,
            camera_matrix,
            np.zeros((4, 1), dtype=np.float64),
            flags=cv2.SOLVEPNP_ITERATIVE
        )

        rotation_matrix = cv2.Rodrigues(rvec)[0]
        return self._rotation_matrix_to_euler(rotation_matrix)

    def _rotation_matrix_to_euler(self, rotation_matrix):
        # 保持符号反转逻辑
        x = math.atan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
        y = math.atan2(-rotation_matrix[2, 0], math.sqrt(rotation_matrix[0, 0]**2 + rotation_matrix[1, 0]**2))
        z = math.atan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
        return np.degrees([x, y, z])