import cv2
from config.settings import CONFIG

class CameraManager:
    def __init__(self, source):
        self.source = source
        self.cap = None
        self._init_opencl()  # 新增OpenCL初始化
        self._init_camera()

    def _init_opencl(self):
        if CONFIG['hardware_acceleration']['enable']:
            if cv2.ocl.haveOpenCL():
                cv2.ocl.setUseOpenCL(True)
                print("OpenCL acceleration enabled")
            else:
                print("OpenCL not available, using CPU")
                CONFIG['hardware_acceleration']['enable'] = False

    def _get_backend(self):
        import platform
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
        """Automatic camera detection logic"""
        # Try user-specified input sources first
        backend = self._get_backend()
        self.cap = self._try_open_source(self.source, backend)
        
        # If that fails, try autodetection
        if not self.cap or not self.cap.isOpened():
            detected_source = self.autodetect_camera_source(backend)
            if detected_source is not None:
                self.source = detected_source
                self.cap = self._try_open_source(detected_source, backend)
            
        if not self.cap.isOpened():
            raise RuntimeError(f"Unable to open video source: {self.source}")

        # Set the resolution
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CONFIG['camera']['width'])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CONFIG['camera']['height'])
        print(f"Camera resolution: {self.width}x{self.height}")

    def _try_open_source(self, source, backend):
        """Try to open the specified input source"""
        try:
            source_int = int(source)
            return cv2.VideoCapture(source_int, backend)
        except ValueError:
            return cv2.VideoCapture(source, backend)

    def autodetect_camera_source(self, backend):
        """Automatically detect available cameras"""
        for i in range(0, 4):  # Detect the first 4 devices
            cap = cv2.VideoCapture(i, backend)
            if cap.isOpened():
                print(f"Auto-detected camera at index {i}")
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
        #return frame if ok else None

    def release(self):
        if self.cap and self.cap.isOpened():
            self.cap.release()

    def _get_backend(self):
        backend_map = {
            'msmf': cv2.CAP_MSMF,
            'v4l2': cv2.CAP_V4L2,
            'dshow': cv2.CAP_DSHOW,
            'auto': cv2.CAP_ANY
        }
        selected = CONFIG['hardware_acceleration'].get('backend', 'auto')
        return backend_map.get(selected, cv2.CAP_ANY)