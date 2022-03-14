# Validation for working cuda devices

# import cupy as cp
import cv2 as cv
import sys

if __name__ == "__main__":
    devices_count = cv.cuda.getCudaEnabledDeviceCount()
    print('Enabled cv2 CUDA devices:', devices_count)
    # print('Enabled cupy CUDA devices:', cp.cuda.runtime.getDeviceCount())
    if devices_count == 0:
        print('CV2 Build location: ', cv.__file__)
        print('CV2 Build info: ', cv.getBuildInformation())
    print('Interpreter:', sys.version)
