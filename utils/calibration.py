import json
from pathlib import Path
from config.settings import CONFIG

def save_calibration(features):
    """保存当前特征值为基准值"""
    calib_data = {
        'mouth_width': features.get('mouth_width', 0),
        'brow_left': features.get('left_brow', 0),
        'brow_right': features.get('right_brow', 0),
        'teeth_open': features.get('teeth_open', 0)
    }
    
    calib_file = Path(CONFIG['calibration']['file'])
    with open(calib_file, 'w') as f:
        json.dump(calib_data, f, indent=2)
    print(f"[校准完成] 基准值已保存至 {calib_file}")