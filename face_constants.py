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
LEFT_PUPIL_IDS = [468, 469, 470, 471, 472]  # 左瞳孔环形关键点
RIGHT_PUPIL_IDS = [473, 474, 475, 476, 477]  # 右瞳孔环形关键点
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
    [285, 528, 200],
    [285, 371, 152],
    [197, 574, 128],
    [173, 425, 108],
    [360, 574, 128],
    [391, 425, 108]
], dtype=np.float64)

# 加载校准
if Path(CALIB_FILE).exists():
    try:
        with open(CALIB_FILE, 'r') as f:
            calib = json.load(f)
    except (json.JSONDecodeError, IOError):
        calib = {}
else:
    calib = {}