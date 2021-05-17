import cv2 
import numpy as np
import subprocess
import sys
import time
import math
from sys import platform
import numpy as np
import os
import flir
from threading import Thread
import random

temperature = (40,20,30)

def saveCurrentTemperature(send_temperature):
    writer = open('temperature.txt', 'w')
    writer.write('%s' % str(send_temperature))
    writer.close()

def getTemp():
    f = flir.Flir()
    global temperature
    maxTemp = float(f.getBox(1)['maxT'].decode("utf-8")[1:-2])
    minTemp = float(f.getBox(1)['minT'].decode("utf-8")[1:-2])
    avgTemp = float(f.getBox(1)['avgT'].decode("utf-8")[1:-2])
    temperature = (maxTemp, minTemp, avgTemp)
    #return temperature

dir_path = os.path.dirname(os.path.realpath(__file__))  
try:
    if platform == "win32":
        sys.path.append(dir_path + '/openpose/python/Release');
        os.environ['PATH']  = os.environ['PATH'] + ';' + dir_path + '/openpose/x64/Release;' +  dir_path + '/openpose/bin;'
        import pyopenpose as op
    else:
        from openpose import pyopenpose as op
except ImportError as e:
    print('Error: OpenPose library could not be found. Did you enable `BUILD_PYTHON` in CMake and have this Python script in the right folder?')
    raise e

fps_time = 0

params = dict()
params["model_folder"] = "openpose/models/"
params["net_resolution"] ="128x128"
opWrapper = op.WrapperPython()  
opWrapper.configure(params)
opWrapper.start()

while True: 
    cap = cv2.VideoCapture('http://admin:admin@169.254.101.173/snapshot.jpg?user=admin&pwd=admin&strm=0') 
 
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    if not ret:
        print("Camera read Error")
        break

    thr1 = Thread(target=getTemp)
    thr1.daemon = True
    thr1.start() 

    datum = op.Datum()
    datum.cvInputData = frame
    fps = 1.0 / (time.time() - fps_time)
    fps_time = time.time()
    opWrapper.emplaceAndPop([datum])
    newImage = datum.cvOutputData.copy()

    if(not (datum.poseKeypoints).shape == ()):
        (maxTemp, minTemp, avgTemp) = temperature
        for keypoints in datum.poseKeypoints:	
            
            x_start = int(keypoints[2][0] + (keypoints[15][0] - keypoints[2][0])/1.5)
            y_start = int(keypoints[0][1] - keypoints[0][1]*0.38)
            x_end = int(keypoints[5][0] - (keypoints[5][0] - keypoints[16][0])/1.5)
            y_end = int(keypoints[0][1])

            start_point = (x_start, y_start)
            end_point = (x_end, y_end)
            cv2.rectangle(frame, start_point, end_point, (255,0,255), 1) 

            value_mean = np.mean(frame[y_start:y_end, x_start:x_end])
            value_mean_norm = value_mean/255
            temp = (maxTemp*value_mean_norm) + (minTemp*(1-value_mean_norm) +2)
            cv2.putText(frame , "%.1f" % (temp), start_point,  cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            t1 = Thread(target=saveCurrentTemperature,args=[temp])
            t1.deamon = True
            t1.start()

    cv2.putText(frame , "FPS: %f" % (fps), (20, 40),  cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("OpenPose 1.6.0", frame)
    #cv2.imshow("OpenPose 1.6.0", newImage)

    if cv2.waitKey(30) & 0x7F == ord('q'):
        print("Exit requested.")
        break
    
cap.release() 
cv2.destroyAllWindows()
