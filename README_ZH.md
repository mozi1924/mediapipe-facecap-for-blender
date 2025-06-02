# Mozi的Blender面部捕捉工具  

[English Document](/README.md)  

## 简介  
基于MediaPipe的面部捕捉程序，主要适配Blender。包含配套的Blender插件（接收端）与传输端。该工具将面部数据转化为简单形变（如位移、缩放）以驱动Blender中的控制骨骼。相较于现有面部捕捉插件，本方案更简洁直观，兼容性更佳。理论上可适配任何遵循相似操作逻辑的面部绑定系统。  

---  

## 功能特性  
- 支持OpenCL硬件加速  
- 基于UDP网络传输  
- 兼容Linux（主要开发环境）  
- 传输端支持CSV录制  
- 无头模式（无需图形界面即可运行）  

---  

## 使用指南  
请从[发布页面](https://github.com/mozi1924/mediapipe-facecap-for-blender/releases)下载适用于您操作系统的版本。  

1. 创建 `_internal/mediapipe/modules` 文件夹  
2. 将 `modules.zip` 解压至该目录  

**目录结构示例**:  
```
~/myapp/main/_internal/mediapipe/modules$ tree
.
├── face_detection
│   ├── face_detection_full_range_cpu.binarypb
│   ├── face_detection_full_range_sparse.tflite
│   ├── face_detection_pb2.py
│   ├── face_detection_short_range_cpu.binarypb
│   └── face_detection_short_range.tflite
├── face_geometry
│   ├── data
│   │   └── __init__.py
│   ├── effect_renderer_calculator_pb2.py
│   ├── env_generator_calculator_pb2.py
│   ├── geometry_pipeline_calculator_pb2.py
│   ├── __init__.py
│   ├── libs
│   │   └── __init__.py
│   └── protos
│       ├── environment_pb2.py
│       ├── face_geometry_pb2.py
│       ├── geometry_pipeline_metadata_pb2.py
│       ├── __init__.py
│       └── mesh_3d_pb2.py
└── face_landmark
    ├── face_landmark_front_cpu.binarypb
    ├── face_landmark.tflite
    ├── face_landmark_with_attention.tflite
    └── __init__.py
```  

### Windows用户注意事项  
**注意：在windows下无头模式无法正常工作**
在Windows系统中，**必须**通过 `--udp_ip` 参数指定有效IP地址（如本地回环地址 `127.0.0.1`），因为不支持 `0.0.0.0`。示例:  
```bash
.\main.exe --preview --input 1 --udp_ip 127.0.0.1
```  

---  

## 手动安装  
**推荐使用Python 3.12**  
```bash
git clone https://github.com/mozi1924/mediapipe-facecap-for-blender.git
python -m venv venv
pip install mediapipe numpy pyyaml
```  

**运行命令**:  
```bash
python main.py --preview --input 1
```  
*(Windows用户需添加 `--udp_ip 127.0.0.1`)*  

---  

## 参数说明  
```bash
usage: main.py [-h] [--input INPUT] [--udp_ip UDP_IP] [--udp_port UDP_PORT] [--preview] [--no_smooth] [--record] [--record_fps RECORD_FPS] [--record_output RECORD_OUTPUT]

选项:
  -h, --help            显示帮助信息
  --input INPUT         视频输入源
  --udp_ip UDP_IP       UDP目标IP（Windows用户必填，本地测试请使用127.0.0.1）
  --udp_port UDP_PORT   UDP端口号
  --preview             启用实时预览
  --no_smooth          禁用平滑处理
  --record             启用CSV录制
  --record_fps RECORD_FPS
                        录制帧率（覆盖配置文件）
  --record_output RECORD_OUTPUT
                        CSV输出路径（默认自动生成）
```  

---  

## 快捷键  
按下 `C` 键进行面部校准,`H`键进行头部校准  （你需要带上`--preview`才能够使用该快捷键）

---  

## 接收端配置  
Blender插件安装说明请查阅 [addons.md](/addons.md)。  

---  

## 示例绑定  
通过[示例骨架](/example/)快速理解原理与工作流程。  

---  

## 许可证  
GPL-3.0  

---  

## 联系方式  
- 邮箱: mozi1924@arasaka.ltd  
- 哔哩哔哩: https://space.bilibili.com/434156493  

---  

## 致谢  
特别感谢 [shenasa-ai/head-pose-estimation](https://github.com/shenasa-ai/head-pose-estimation) 对头部旋转算法的贡献。  
