import cv2
import platform
from config.settings import CONFIG

class CameraManager:
    def __init__(self, source):
        self.source = source
        self.cap = None
        self._init_opencl()
        self._init_camera()
        self._detect_best_settings()  # 新增自动检测逻辑

    def _init_opencl(self):
        """初始化OpenCL加速"""
        if CONFIG['hardware_acceleration']['enable']:
            if cv2.ocl.haveOpenCL():
                cv2.ocl.setUseOpenCL(True)
                print("OpenCL acceleration enabled")
            else:
                print("OpenCL not available, using CPU")
                CONFIG['hardware_acceleration']['enable'] = False

    def _get_backend(self):
        """获取平台对应的视频后端"""
        system = platform.system()
        backend_map = {
            'Windows': {
                'auto': cv2.CAP_DSHOW,
                'msmf': cv2.CAP_MSMF,
                'dshow': cv2.CAP_DSHOW
            },
            'Linux': {
                'auto': cv2.CAP_V4L2,
                'v4l2': cv2.CAP_V4L2
            },
            'Darwin': {
                'auto': cv2.CAP_AVFOUNDATION,
                'avfoundation': cv2.CAP_AVFOUNDATION
            }
        }
        selected = CONFIG['hardware_acceleration'].get('backend', 'auto')
        return backend_map[system].get(selected, backend_map[system]['auto'])

    def _init_camera(self):
        """初始化摄像头设备"""
        backend = self._get_backend()
        self.cap = self._try_open_source(self.source, backend)
        
        if not self.cap or not self.cap.isOpened():
            detected_source = self.autodetect_camera_source(backend)
            if detected_source is not None:
                self.source = detected_source
                self.cap = self._try_open_source(detected_source, backend)
            
        if not self.cap.isOpened():
            raise RuntimeError(f"无法打开视频源: {self.source}")

        # 初始分辨率设置（后续会被覆盖）
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    def _detect_best_settings(self):
        """自动检测最佳摄像头设置"""
        # 1. 尝试首选视频格式
        preferred_format = CONFIG['camera'].get('preferred_format', 'MJPG')
        if self._try_set_format(preferred_format):
            print(f"成功设置首选格式: {preferred_format}")
        else:
            print(f"无法设置首选格式 {preferred_format}，尝试其他格式...")
            for fmt in ['YUYV', 'H264', 'NV12', 'YV12']:
                if self._try_set_format(fmt):
                    print(f"回退使用格式: {fmt}")
                    break
        
        # 2. 设置分辨率
        if CONFIG['camera']['width'] == 'auto' or CONFIG['camera']['height'] == 'auto':
            self._set_optimal_resolution()
        else:
            self._set_manual_resolution()

        # 最终验证
        print(f"视频格式: {self._get_fourcc()}")
        print(f"最终分辨率: {self.width}x{self.height}")

    def _try_set_format(self, fourcc):
        """尝试设置指定视频格式"""
        try:
            code = cv2.VideoWriter_fourcc(*fourcc)
            self.cap.set(cv2.CAP_PROP_FOURCC, code)
            return fourcc == self._get_fourcc()
        except:
            return False

    def _get_fourcc(self):
        """获取当前视频格式"""
        code = int(self.cap.get(cv2.CAP_PROP_FOURCC))
        return "".join([chr((code >> 8*i) & 0xFF) for i in range(4)]).strip()

    def _set_optimal_resolution(self):
        """自动设置最佳分辨率"""
        resolutions = [
            (3840, 2160), (2560, 1440),
            (1920, 1080), (1280, 720),
            (640, 480), (320, 240)
        ]
        
        for w, h in resolutions:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
            if self.width == w and self.height == h:
                print(f"自动选择分辨率: {w}x{h}")
                return
        
        # 保底设置
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        print(f"使用默认分辨率: 640x480")

    def _set_manual_resolution(self):
        """设置手动指定分辨率"""
        target_w = CONFIG['camera']['width']
        target_h = CONFIG['camera']['height']
        
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, target_w)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, target_h)
        
        if self.width != target_w or self.height != target_h:
            print(f"警告: 无法设置 {target_w}x{target_h}，实际分辨率 {self.width}x{self.height}")

    def _try_open_source(self, source, backend):
        """尝试打开视频源"""
        try:
            source_int = int(source)
            return cv2.VideoCapture(source_int, backend)
        except ValueError:
            return cv2.VideoCapture(source, backend)

    def autodetect_camera_source(self, backend):
        """自动检测可用摄像头"""
        for i in range(0, 4):
            cap = cv2.VideoCapture(i, backend)
            if cap.isOpened():
                print(f"自动检测到摄像头索引: {i}")
                return i
            cap.release()
        return None

    @property
    def width(self):
        return int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    @property
    def height(self):
        return int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def read_frame(self):
        ok, frame = self.cap.read()
        return cv2.flip(frame, 1) if ok else None

    def release(self):
        if self.cap and self.cap.isOpened():
            self.cap.release()