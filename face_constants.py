# face_constants.py
import json
import numpy as np
from pathlib import Path
from config.settings import CONFIG
import os
import time

# 使用带时间戳的缓存
_calib_cache = {"data": {}, "mtime": 0}
_head_calib_cache = {"data": {}, "mtime": 0}

def get_calib():
    """带智能缓存的面部校准数据"""
    global _calib_cache
    calib_file = Path(CALIB_FILE)
    
    # 检查文件是否存在并获取修改时间
    current_mtime = calib_file.stat().st_mtime if calib_file.exists() else 0
    
    # 如果文件已修改或缓存为空，重新加载
    if current_mtime > _calib_cache["mtime"] or not _calib_cache["data"]:
        try:
            with open(calib_file, 'r') as f:
                _calib_cache["data"] = json.load(f)
            _calib_cache["mtime"] = current_mtime
        except (FileNotFoundError, json.JSONDecodeError):
            _calib_cache["data"] = {}
    
    return _calib_cache["data"]

def get_head_calib():
    """带智能缓存的头部校准数据"""
    global _head_calib_cache
    calib_file = Path(HEAD_CALIB_FILE)
    
    # 检查文件是否存在并获取修改时间
    current_mtime = calib_file.stat().st_mtime if calib_file.exists() else 0
    
    # 如果文件已修改或缓存为空，重新加载
    if current_mtime > _head_calib_cache["mtime"] or not _head_calib_cache["data"]:
        try:
            with open(calib_file, 'r') as f:
                _head_calib_cache["data"] = json.load(f)
            _head_calib_cache["mtime"] = current_mtime
        except (FileNotFoundError, json.JSONDecodeError):
            _head_calib_cache["data"] = {}
    
    return _head_calib_cache["data"]

# 初始化路径
CALIB_FILE = CONFIG['calibration']['file']
HEAD_CALIB_FILE = CONFIG['head_calibration']['file']

# 添加参考点常量
REF_POINTS = CONFIG['calibration'].get('ref_points', [0, 1])
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