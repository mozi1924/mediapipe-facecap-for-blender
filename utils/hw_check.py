import cv2
import platform

def print_hw_info():
    print(f"Platform: {platform.platform()}")
    #print(f"OpenCV build info:\n{cv2.getBuildInformation()}")
    print(f"OpenCL available: {cv2.ocl.haveOpenCL()}")
    if cv2.ocl.haveOpenCL():
        print(f"OpenCL device: {cv2.ocl.Device_getDefault().name()}")