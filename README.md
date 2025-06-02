# Mozi's Facemocap for Blender

[中文文档](/README_ZH.md)

## Introduction  
A MediaPipe-based facial capture program primarily designed for Blender, consisting of a companion Blender addon (referred to as the "receiver") and a transmitter. It converts facial data into simple transformations (e.g., movement, scaling) for control bones in Blender. Compared to existing facial capture plugins, it is simpler, more intuitive, and offers better compatibility. Theoretically, it works with any facial rigging system that follows similar operational logic.

---

## Features  
- OpenCL hardware acceleration support  
- UDP-based network transmission  
- Linux compatibility (developed primarily on this OS)  
- CSV recording support in the transmitter  
- Headless mode (runs without a GUI)  

---

## Usage  
Download the version appropriate for your operating system from the [releases page](https://github.com/mozi1924/mediapipe-facecap-for-blender/releases).  

1. Create the `_internal/mediapipe/modules` folder.  
2. Unzip `modules.zip` into this folder.  

**Folder Structure**:  
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

### Windows Users Note  
**Note: Headless mode does not work properly under Windows**
On Windows, you **must** specify the `--udp_ip` parameter with a valid IP address (e.g., `127.0.0.1` for localhost), as `0.0.0.0` is not supported. Example:  
```bash
.\main.exe --preview --input 1 --udp_ip 127.0.0.1
```  

---

## Manual Installation  
**Python 3.12 is recommended.**  
```bash
git clone https://github.com/mozi1924/mediapipe-facecap-for-blender.git
python -m venv venv
pip install mediapipe numpy pyyaml
```  

**Run**:  
```bash
python main.py --preview --input 1
```  
*(Add `--udp_ip 127.0.0.1` on Windows)*  

---

## Parameters  
```bash
Mozi's Facecap Transmitter

options:
  -h, --help            show this help message and exit
  --input INPUT         Video source (auto for detection)
  --udp_ip UDP_IP       UDP Destination IP
  --udp_port UDP_PORT   UDP port
  --preview             Enable Live Preview
  --no_smooth           Disable Smoothing
  --record              Enable CSV Recording
  --record_fps RECORD_FPS
                        Override recording FPS
  --camera_config CAMERA_CONFIG
                        Custom camera config file

```  

---

## Shortcut Key  
Press `C` to calibrate the face,`H` to calibrate the head (You need to use `--preview` to use this shortcut)

---

## Receiver  
Please read [addons.md](/addons.md) for Blender addon setup.  

---

## Example Rig  
Try the [example rig](/example/) to quickly understand the principles and workflow.  

---

## License  
GPL-3.0  

---

## Contact  
- Email: mozi1924@arasaka.ltd  
- BiliBili: https://space.bilibili.com/434156493  

---

## Acknowledgments  
Special thanks to [shenasa-ai/head-pose-estimation](https://github.com/shenasa-ai/head-pose-estimation) for solving head rotation challenges.