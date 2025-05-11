import json
from pathlib import Path
from config.settings import CONFIG

def save_calibration(features):
    """Save the current feature value as the reference value"""
    calib_data = {
        'mouth_width': features.get('mouth_width', 0),
        'brow_left': features.get('left_brow', 0),
        'brow_right': features.get('right_brow', 0),
        'teeth_open': features.get('teeth_open', 0)
    }
    
    calib_file = Path(CONFIG['calibration']['file'])
    with open(calib_file, 'w') as f:
        json.dump(calib_data, f, indent=2)
    print(f"[Calibration Completed] Reference values ​​have been saved to {calib_file}")