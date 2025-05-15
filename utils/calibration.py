import json
from pathlib import Path
from config.settings import CONFIG
from models.face_utils import save_head_calibration

def save_calibration(raw_features):
    # 统一校准入口，处理面部和头部
    calib_data = {
        'mouth_width': raw_features.get('_raw_mouth_width', 0),
        'brow_left': raw_features.get('_raw_left_brow', 0),
        'brow_right': raw_features.get('_raw_right_brow', 0),
        'teeth_open': raw_features.get('_raw_teeth_open', 0)
    }
    save_head_calibration(raw_features)  # 调用头部校准
    
    with open(CONFIG['calibration']['file'], 'w') as f:
        json.dump(calib_data, f, indent=2)
    print(f"[Calibration Completed] Reference values ​​have been saved to {CONFIG['calibration']['file']}")