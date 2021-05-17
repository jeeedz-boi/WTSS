import cv2 
import numpy as np
import subprocess

import urllib.request 

import sys
import time
import math
from sys import platform
import numpy as np
import os
import flir
# import flir_image_extractor
from threading import Thread

class temperatureRetriver ():
    def __init__(self):
        self.temperature = (60.0,20.0)

    def getTempFromFlir(self):
        f = flir.Flir()
        maxTemp = float(f.getBox(2)['maxT'].decode("utf-8")[1:-2])
        minTemp = float(f.getBox(2)['minT'].decode("utf-8")[1:-2])
        self.temperature = (maxTemp, minTemp)
    
    def getTemperature(self, start_x, start_y, end_x, end_y):
        self.getTempFromFlir()

        req = urllib.request.urlopen('http://169.254.101.173/snapshot.jpg?user=admin&pwd=admin&strm=0') 
        arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
        img = cv2.imdecode(arr, -1)

        value_mean = np.mean(img[start_y:end_y, start_x:end_x])
        value_mean_nom = value_mean/255

        temp = (maxTemp*value_mean_nom)+(minTemp*(1-value_mean_nom))

        return temp


