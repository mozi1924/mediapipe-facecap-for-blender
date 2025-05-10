import argparse
import cv2
import mediapipe as mp
import time
import numpy as np
from pathlib import Path
from config.settings import CONFIG
from utils.face_utils import calculate_features, draw_preview, MODEL_POINTS
from utils.network import UDPTransmitter
from models.smoother import FeatureSmoother
from utils.calibration import save_calibration

# ----------------------------
# 参数解析模块
# ----------------------------
def parse_args():
    parser = argparse.ArgumentParser(description='面部特征捕捉发射端')
    parser.add_argument('--input', type=str, default='0', help='视频输入源')
    parser.add_argument('--udp_ip', type=str, default='0.0.0.0', help='UDP目标IP')
    parser.add_argument('--udp_port', type=int, default=12345, help='UDP端口号')
    parser.add_argument('--preview', action='store_true', help='启用实时预览')
    parser.add_argument('--no_smooth', action='store_true', help='禁用平滑')
    return parser.parse_args()

# ----------------------------
# 摄像头管理类
# ----------------------------
class CameraManager:
    def __init__(self, source):
        self.source = source
        self.cap = None
        self._init_camera()

    def _init_camera(self):
        """初始化摄像头硬件参数"""
        backend = self._get_backend()
        self.cap = cv2.VideoCapture(self.source, backend)
        
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(int(self.source))  # 回退尝试
            
        if not self.cap.isOpened():
            raise RuntimeError(f"无法打开视频源：{self.source}")

        # 设置分辨率
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CONFIG['camera']['width'])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CONFIG['camera']['height'])
        print(f"摄像头分辨率: {self.width}x{self.height}")

    def _get_backend(self):
        """获取视频捕获后端"""
        backend_map = {
            'msmf': cv2.CAP_MSMF,
            'v4l2': cv2.CAP_V4L2,
            'dshow': cv2.CAP_DSHOW,
            'auto': cv2.CAP_ANY
        }
        selected = CONFIG['hardware_acceleration'].get('backend', 'auto')
        return backend_map.get(selected, cv2.CAP_ANY)

    @property
    def width(self):
        return int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    @property
    def height(self):
        return int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def read_frame(self):
        """读取并预处理帧"""
        ok, frame = self.cap.read()
        if not ok: return None
        return cv2.flip(frame, 1)  # 镜像处理

    def release(self):
        if self.cap.isOpened():
            self.cap.release()

# ----------------------------
# 面部检测器类
# ----------------------------
class FaceMeshDetector:
    def __init__(self):
        self.face_mesh = self._init_face_mesh()

    def _init_face_mesh(self):
        """根据配置初始化MediaPipe"""
        return mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def process(self, frame):
        """处理帧并返回特征点"""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return self.face_mesh.process(rgb)

    def close(self):
        self.face_mesh.close()

# ----------------------------
# 主程序逻辑
# ----------------------------
def main():
    args = parse_args()
    
    # 初始化组件
    camera = CameraManager(args.input)
    detector = FaceMeshDetector()
    transmitter = UDPTransmitter(args.udp_ip, args.udp_port)
    smoother = FeatureSmoother() if not args.no_smooth else None

    last_send = 0
    try:
        while True:
            frame = camera.read_frame()
            if frame is None: break

            # 特征检测
            res = detector.process(frame)
            if not res.multi_face_landmarks: continue

            # 计算特征
            lm = res.multi_face_landmarks[0].landmark
            features = calculate_features(lm, frame.shape)

            # 平滑处理
            if smoother:
                features = smoother.apply(features)

            # 按配置FPS发送数据
            if time.time() - last_send > 1/CONFIG['preview']['fps']:
                transmitter.send(features)
                last_send = time.time()

            # 实时预览
            if args.preview:
                preview_img = draw_preview(frame.copy(), features)
                cv2.imshow('Preview', preview_img)
                if cv2.waitKey(1) & 0xFF == 27:  # ESC退出
                    break
            
            # main.py的主循环中添加
            if args.preview:
                preview_img = draw_preview(frame.copy(), features)
                cv2.imshow('Preview', preview_img)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('c'):  # 按下C键触发校准
                    save_calibration(features)  # 调用校准函数

    finally:
        camera.release()
        detector.close()
        transmitter.close()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()