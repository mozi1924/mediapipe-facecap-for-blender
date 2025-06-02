# calibration.py
import json
from config.settings import CONFIG

def save_calibration(raw_features):
    """保存面部校准数据"""
    calib_data = {
        'mouth_width': raw_features.get('_raw_mouth_width', 0),
        'brow_left': raw_features.get('_raw_left_brow', 0),
        'brow_right': raw_features.get('_raw_right_brow', 0),
        'teeth_open': raw_features.get('_raw_teeth_open', 0)
    }
    
    with open(CONFIG['calibration']['file'], 'w') as f:
        json.dump(calib_data, f, indent=2)
    print(f"Facial calibration saved to {CONFIG['calibration']['file']}")

def save_head_calibration(raw_features):
    """保存头部校准数据"""
    calib_data = {
        'pitch': raw_features.get('_raw_head_pitch', 0),
        'yaw': raw_features.get('_raw_head_yaw', 0),
        'roll': raw_features.get('_raw_head_roll', 0)
    }
    
    with open(CONFIG['head_calibration']['file'], 'w') as f:
        json.dump(calib_data, f, indent=2)
    print(f"Head calibration saved to {CONFIG['head_calibration']['file']}")

def reset_calibration():
    """重置所有校准数据"""
    # 重置面部校准
    default_calib = {
        'mouth_width': 0,
        'brow_left': 0,
        'brow_right': 0,
        'teeth_open': 0
    }
    with open(CONFIG['calibration']['file'], 'w') as f:
        json.dump(default_calib, f, indent=2)
    
    # 重置头部校准
    default_head_calib = {
        'pitch': 0,
        'yaw': 0,
        'roll': 0
    }
    with open(CONFIG['head_calibration']['file'], 'w') as f:
        json.dump(default_head_calib, f, indent=2)
    
    print("All calibration data has been reset")