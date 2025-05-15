import json
from pathlib import Path
from config.settings import CONFIG
from models.face_utils import save_head_calibration

def save_calibration(raw_features):
    """直接使用原始值进行校准"""
    calib_data = {
        'mouth_width': raw_features.get('_raw_mouth_width', 0),
        'brow_left': raw_features.get('_raw_left_brow', 0),
        'brow_right': raw_features.get('_raw_right_brow', 0),
        'teeth_open': raw_features.get('_raw_teeth_open', 0)
    }
    
    save_head_calibration(raw_features)  # 假设头部校准也需要原始值

    calib_file = Path(CONFIG['calibration']['file'])
    with open(calib_file, 'w') as f:
        json.dump(calib_data, f, indent=2)
    print(f"[Calibration Completed] 基准值已保存至 {calib_file}")