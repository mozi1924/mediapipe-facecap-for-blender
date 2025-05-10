import os
import yaml
from pathlib import Path

CONFIG_FILE = 'config.yaml'  # 新增这一行

DEFAULT_CONFIG = {
    'hardware_acceleration': {
        'enable': False,
        'opencl': False,
        'backend': 'auto'
    },
    'smoothing': {
        'enable': True,
        'head': 0.8,
        'eyelids': 0.6,
        'pupils': 0.3,
        'mouth': 0.7,
        'brows': 0.5,
        'teeth': 0.4
    },
    'calibration': {
        'file': 'calibration.json',
        'ref_points': [33, 263]
    },
    'recording': {
        'fps': 30,
        'output_dir': 'recordings',
        'auto_timestamp': True
    },
    'preview': {
        'fps': 30,
        'scale': 0.8
    },
    'camera': {
        'width': 1280,
        'height': 720
    }
}

def load_config():
    config_path = Path(CONFIG_FILE)
    if not config_path.exists():
        config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(config_path, 'w') as f:
            yaml.safe_dump(DEFAULT_CONFIG, f, sort_keys=False)
        return DEFAULT_CONFIG.copy()
    with open(config_path, 'r') as f:
        user_cfg = yaml.safe_load(f)
    merged = DEFAULT_CONFIG.copy()
    merged.update(user_cfg or {})
    return merged

CONFIG = load_config()