# Mozi's facemocap for blender

## 介绍
只是一个基于MediaPipe的面部捕捉程序主要用于blender，它有一个配套的blender插件（以下简称接收端）和一个发射端

它用于将面部信息转换成blender中控制器骨骼的简单变换，比如移动，缩放等，相比现有的面捕插件更加简单易懂，兼容性更强，理论上来说，它适用于任何操作逻辑类似的面部绑定

## 特性
- 支持OpenCL硬件加速
- 使用UDP进行网络传输
- 支持Linux操作系统（因为本身就是在这上面进行开发的）
- 发射端支持录制csv文件
- 无头模式（可以在无图形界面的情况下运行）

## 安装
建议使用python3.12
```
git clone https://github.com/mozi1924/mediapipe-facecap-for-blender.git

python -m venv venv

pip install mediapipe numpy pyyaml
```

## 运行
```
python main.py --preview --input 1
```
可能需要根据实际情况进行修改

## 参数
```
[mozi@navi]~/mediapipe-realtime-face/app-next% python main.py --help             
usage: main.py [-h] [--input INPUT] [--udp_ip UDP_IP] [--udp_port UDP_PORT] [--preview] [--no_smooth] [--record]
               [--record_fps RECORD_FPS] [--record_output RECORD_OUTPUT]

面部特征捕捉发射端

options:
  -h, --help            show this help message and exit
  --input INPUT         视频输入源
  --udp_ip UDP_IP       UDP目标IP
  --udp_port UDP_PORT   UDP端口号
  --preview             启用实时预览
  --no_smooth           禁用平滑
  --record              启用录制到CSV
  --record_fps RECORD_FPS
                        录制帧率（覆盖配置文件）
  --record_output RECORD_OUTPUT
                        CSV输出路径（默认根据时间生成）
```

## 接收端
请阅读 addons.md[addons.md](./addons.md)

## LICENSE
GPL-3.0

## 联系
Email mozi1924@arasaka.ltd
BiliBili https://space.bilibili.com/434156493