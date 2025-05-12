# face_constants.py
import json
import cv2
import math
import numpy as np
from pathlib import Path
from config.settings import CONFIG

# --------------------------
# 全局配置
# --------------------------
CALIB_FILE = CONFIG['calibration']['file']

# --------------------------
# 面部关键点定义
# --------------------------
# 嘴唇相关
LIPS_UP = [13]
LIPS_DOWN = [14]
LEFT_LIP_CORNER = 61
RIGHT_LIP_CORNER = 291
MOUTH_LEFT = 61
MOUTH_RIGHT = 291
MOUTH_LOWER_CENTER = 164

# 眼睛相关
LEFT_EYE_UP = [159]
LEFT_EYE_DOWN = [145]
RIGHT_EYE_UP = [386]
RIGHT_EYE_DOWN = [374]
LEFT_PUPIL = 468
RIGHT_PUPIL = 473
LEFT_EYE = [33, 133]    # 左眼内外角
RIGHT_EYE = [263, 362]  # 右眼内外角
LEFT_EYE_INNER = 133
LEFT_EYE_OUTER = 33
RIGHT_EYE_INNER = 362
RIGHT_EYE_OUTER = 263

# 眉毛相关
BROW_CENTER_LEFT = 107
BROW_CENTER_RIGHT = 336
LEFT_BROW_IDS = [65, 55, 52, 53, 46]
RIGHT_BROW_IDS = [295, 285, 282, 283, 276]

# 头部方向
NOSE_TIP = 1
CHIN = 152

# --------------------------
# 3D模型点 (供头部旋转计算使用)
# --------------------------
MODEL_POINTS = np.array([
    (0.0, 0.0, 0.0),         # NOSE_TIP
    (0.0, -330.0, -65.0),    # CHIN
    (-165.0, 170.0, -135.0), # LEFT_EYE_OUTER
    (165.0, 170.0, -135.0),  # RIGHT_EYE_OUTER 
    (-75.0, -40.0, -125.0),  # MOUTH_LEFT
    (75.0, -40.0, -125.0),   # MOUTH_RIGHT
    (-60.0, 130.0, -110.0),  # BROW_CENTER_LEFT
    (60.0, 130.0, -110.0)    # BROW_CENTER_RIGHT
], dtype=np.float64)  # 明确指定数据类型和维度

# 加载校准
CALIB_FILE = CONFIG['calibration']['file']
if Path(CALIB_FILE).exists():
    with open(CALIB_FILE, 'r') as f:
        calib = json.load(f)
else:
    calib = {}