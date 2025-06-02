import argparse
import cv2
import mediapipe as mp
import time
import traceback
from config.settings import CONFIG
from utils.camera import CameraManager
from models.face_utils import calculate_features, draw_preview, save_calibration, save_head_calibration
from utils.network import UDPTransmitter
from models.smoother import FeatureSmoother
from utils.recording import Recorder
from utils.hw_check import print_hw_info

print_hw_info()

def parse_args():
    parser = argparse.ArgumentParser(description="Mozi's Facecap Transmitter")
    parser.add_argument('--input', type=str, default='auto', help='Video source (auto for detection)')
    parser.add_argument('--udp_ip', type=str, default='127.0.0.1', help='UDP Destination IP')
    parser.add_argument('--udp_port', type=int, default=12345, help='UDP port')
    parser.add_argument('--preview', action='store_true', help='Enable Live Preview')
    parser.add_argument('--no_smooth', action='store_true', help='Disable Smoothing')
    parser.add_argument('--record', action='store_true', help='Enable CSV Recording')
    parser.add_argument('--record_fps', type=float, default=None, help='Override recording FPS')
    parser.add_argument('--camera_config', type=str, default=None, help='Custom camera config file')
    return parser.parse_args()

class FaceMeshDetector:
    def __init__(self):
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
    
    def close(self):  # 添加的 close 方法
        self.face_mesh.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()  # 调用 close 方法

    def process(self, frame):
        if CONFIG['hardware_acceleration']['enable']:
            frame_umat = cv2.UMat(frame)
            rgb = cv2.cvtColor(frame_umat, cv2.COLOR_BGR2RGB)
            rgb = cv2.UMat.get(rgb)
        else:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return self.face_mesh.process(rgb)

def main():
    args = parse_args()
    
    camera, detector, transmitter, smoother, recorder = None, None, None, None, None
    
    try:
        camera = CameraManager(args.input)
        detector = FaceMeshDetector()
        transmitter = UDPTransmitter(args.udp_ip, args.udp_port)
        
        if not args.no_smooth:
            smoother = FeatureSmoother()
        if args.record:
            recorder = Recorder(args.record_output, args.record_fps)
        
        print(f"Camera initialized: {camera.width}x{camera.height}")
        
        last_send = 0
        frame_counter = 0
        start_time = time.time()
        
        while True:
            frame = camera.read_frame()
            if frame is None: 
                print("End of video stream")
                break

            res = detector.process(frame)
            if not res.multi_face_landmarks: 
                if args.preview:
                    cv2.imshow('Preview', frame)
                continue

            lm = res.multi_face_landmarks[0].landmark
            features, raw_features = calculate_features(lm, frame.shape)

            if smoother:
                features = smoother.apply(features)

            current_time = time.time()
            if current_time - last_send > 1/CONFIG['preview']['fps']:
                transmitter.send(features)
                last_send = current_time

            if recorder:
                recorder.record(features)

            if args.preview:
                preview_img = draw_preview(frame.copy(), features, lm)
                
                frame_counter += 1
                elapsed = time.time() - start_time
                if elapsed > 1:
                    fps = frame_counter / elapsed
                    cv2.putText(preview_img, f"FPS: {fps:.1f}", 
                               (10, preview_img.shape[0] - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    frame_counter = 0
                    start_time = time.time()
                
                cv2.imshow('Preview', preview_img)
                key = cv2.waitKey(1)
                if key == 27:  # ESC
                    break
                elif key == ord('c'):  # 面部校准
                    save_calibration(raw_features)
                elif key == ord('h'):  # 头部校准
                    save_head_calibration(raw_features)
    
    except Exception as e:
        print(f"Fatal error: {str(e)}")
        traceback.print_exc()
    
    finally:
        if recorder:
            recorder.close()
        if transmitter:
            transmitter.close()
        if detector:
            detector.close()  # 现在可以安全调用
        if camera:
            camera.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()