from read_thermal import flir_image_extractor
import logging
import sys
import time
import math
import cv2
from sys import platform
import numpy as np
import os


if __name__ == '__main__':
    dir_path = os.path.dirname(os.path.realpath(__file__))  
    try:
        if platform == "win32":
            # Change these variables to point to the correct folder (Release/x64 etc.)
            sys.path.append(dir_path + '/python/Release');
            os.environ['PATH']  = os.environ['PATH'] + ';' + dir_path + '/x64/Release;' +  dir_path + '/bin;'
            import pyopenpose as op
        else:
            from openpose import pyopenpose as op
    except ImportError as e:
        print('Error: OpenPose library could not be found. Did you enable `BUILD_PYTHON` in CMake and have this Python script in the right folder?')
        raise e
    
    fps_time = 0

    params = dict()
    params["model_folder"] = "models/"
    params["net_resolution"] ="256x128"
    # params["net_resolution"] ="256x256"
    # params["net_resolution"] ="128x128"
    # params["net_resolution"] ="-1x368"

    # Starting OpenPose
    opWrapper = op.WrapperPython()
    opWrapper.configure(params)
    opWrapper.start()
    print("OpenPose start")
    cap = cv2.VideoCapture(0)
    while (True):
        ret, frame = cap.read()
        # frame = cv2.flip(frame, 1)
        if not ret:
            print("Camera read Error")
            break
        datum = op.Datum()
        datum.cvInputData = frame
        fps = 1.0 / (time.time() - fps_time)
        fps_time = time.time()


        opWrapper.emplaceAndPop(op.VectorDatum([datum]))
        newImage = datum.cvOutputData.copy()
        cv2.putText(newImage , "FPS: %f" % (fps), (20, 40),  cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("OpenPose 1.7.0 - Tutorial Python API", newImage)

        if cv2.waitKey(30) & 0x7F == ord('q'):
            print("Exit requested.")
            break

    cap.release()
    cv2.destroyAllWindows()