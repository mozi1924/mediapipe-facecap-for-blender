import csv
import time
from pathlib import Path
from datetime import datetime
from config.settings import CONFIG

class Recorder:
    def __init__(self, output_path=None, fps=None):
        self.output_path = self._resolve_output_path(output_path)
        self.fps = fps or CONFIG['recording']['fps']
        self.interval = 1.0 / self.fps
        
        self.last_write = 0
        self.start_time = time.time()
        self.file = None
        self.writer = None
        
        # 初始化时立即创建文件
        self._init_csv()
        self.recording_start_time = time.time()

    def _resolve_output_path(self, user_path):
        """Enhanced path processing logic"""
        try:
            if user_path:
                path = Path(user_path)
                if path.suffix == '':  # The user specifies a directory
                    path.mkdir(parents=True, exist_ok=True)
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    return path / f"recording_{timestamp}.csv"
                return path
            
            # Handling default paths
            default_dir = Path(CONFIG['recording'].get('output_dir', 'recordings'))
            default_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            return default_dir / f"recording_{timestamp}.csv"
        except Exception as e:
            raise RuntimeError(f"路径解析失败: {str(e)}")

    def _init_csv(self):
        """Enhanced file initialization logic"""
        try:
            self.file = open(self.output_path, 'w', newline='')
            self.writer = csv.writer(self.file)
            headers = [
                'timestamp',
                'head_pitch', 'head_yaw', 'head_roll',
                'mouth_open', 'mouth_width',
                'left_eyelid', 'right_eyelid',
                'left_pupil_x', 'left_pupil_y',
                'right_pupil_x', 'right_pupil_y'
            ]
            self.writer.writerow(headers)
            print(f"Recording started: {self.output_path}")
            return True
        except IOError as e:
            print(f"File creation failed: {str(e)}")
            return False
        except Exception as e:
            print(f"Initialization Exception: {str(e)}")
            return False

    def record(self, features):
        """Adding null protection"""
        if not self.writer:
            return
            
        current_time = time.time()
        if (current_time - self.last_write) < self.interval:
            return
        
        try:
            elapsed = round(current_time - self.recording_start_time, 3)
            row = [
                elapsed,
                features.get('head_pitch', 0),
                features.get('head_yaw', 0),
                features.get('head_roll', 0),
                features.get('mouth_open', 0),
                features.get('mouth_width', 0),
                features.get('left_eyelid', 0),
                features.get('right_eyelid', 0),
                features.get('left_pupil_x', 0),
                features.get('left_pupil_y', 0),
                features.get('right_pupil_x', 0),
                features.get('right_pupil_y', 0)
            ]
            self.writer.writerow(row)
            self.last_write = current_time
        except Exception as e:
            print(f"Write failed: {str(e)}")

    def close(self):
        """Safety Shutdown"""
        if self.file and not self.file.closed:
            self.file.close()
            self.writer = None
            print(f"Recording saved: {self.output_path}")