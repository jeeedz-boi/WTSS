import os
import cv2
import time
import face_recognition
import pickle
from sklearn import neighbors
import time
from threading import Thread
import numpy as np
import requests
import json
from tensorflow.contrib.keras import applications, preprocessing, models
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
current_milli_time = lambda: int(round(time.time() * 1000))

class Detection():
    def __init__(self):
        self.infoList = dict()
        self.passFlag = False
        self.flag = False
        self.faceNet = self.load_face_detector()
        self.knn_clf = pickle.load(open(os.path.join('models/trained_knn_model.clf'), 'rb'))
        self.fps_time = 0
    
    def send_info_to_database(self, name,temp):
        url = "https://better-wfss-dashboard.herokuapp.com/user"
        name="Thongchai Yamsuk"
        payload = {
           "name": str(name),
           "temp": float(temp) 
        }
        headers = {"content-type": "application/json"}
        r = requests.put(url, data=json.dumps(payload), headers=headers)
        print(r.status_code, name, temp)

    def get_temperature(self):
        reader = open('temperature.txt', 'r')
        #retrieve_temperature = reader.readline()
        retrieve_temperature = 36.99
        reader.close()
        return float(retrieve_temperature)
	
    def load_face_detector(self):
        # load our serialized face detector model from disk
        prototxtPath = r"models/face_detector/deploy.prototxt"
        weightsPath = r"models/face_detector/res10_300x300_ssd_iter_140000.caffemodel"
        faceNet = cv2.dnn.readNetFromCaffe(prototxtPath, weightsPath)
        return faceNet   

    def detect_face(self, frame, faceNet):
        # grab the dimensions of the frame and then construct a blob
        # from it
        (h, w) = frame.shape[:2]
        #blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224),
        #    (104.0, 177.0, 123.0))
        
        blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224),(127,127,127))

        # pass the blob through the network and obtain the face detections
        faceNet.setInput(blob)
        detections = faceNet.forward()

        # initialize our list of faces, their corresponding locations,
        # and the list of predictions from our face mask network
        faces = []
        locs = []
        
        # loop over the detections
        for i in range(0, detections.shape[2]):
            # extract the confidence (i.e., probability) associated with
            # the detection
            confidence = detections[0, 0, i, 2]

            # filter out weak detections by ensuring the confidence is
            # greater than the minimum confidence
            if confidence > 0.5:
                # compute the (x, y)-coordinates of the bounding box for
                # the object
                box = detections[0, 0, i, 3:7]
		# ensure the bouding boxes fall within the dimensions of the frame
                box = [max(0,box[0]), max(0,box[1]), min(1, box[2]), min(1, box[3])]
                # due to the old shit tensorflow it could not provide precise location of faces 
		# so the next if else will scale that location to be mpre resonble  
                if (box[2] <= 0.60): # on left side
                    print('left')
                    scaler = (1-box[2])/5.25
                    box[0] += scaler
                    box[2] += scaler/4
                elif (box[0] >= 0.40): # on right side
                    print('right')
                    scaler = (box[0])/5.25
                    box[0] -= scaler
                    box[2] -= scaler

                # ensure the bouding boxes fall within the dimensions of the frame again
                box = [max(0,box[0]), max(0,box[1]), min(1, box[2]), min(1, box[3])] 
                box = box * np.array([w, h, w, h])

                (startX, startY, endX, endY) = box.astype("int")
                # ensure the bounding boxes fall within the dimensions of the frame
                (startX, startY) = (max(0, startX), max(0, startY))
                (endX, endY) = (min(w - 1, endX), min(h - 1, endY))
                
                # extract the face ROI, convert it from BGR to RGB channel
                # ordering, resize it to 224x224, and preprocess it
                face = frame[startY:endY, startX:endX]
                face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                face = cv2.resize(face, (224, 224))
                cv2.imshow('cropped', face)
                #face = preprocessing.image.img_to_array(face)
                #face = applications.xception.preprocess_input(face)

                # add the face and bounding boxes to their respective
                # lists`
                #faces.append(face)
                locs.append((startX, startY, endX, endY))
        return locs

    def face_rec(self, X_face_locations, rgb_small_frame, distance_threshold=0.5):
        faces_encodings = face_recognition.face_encodings(rgb_small_frame, known_face_locations=X_face_locations)
        # Use the KNN model to find the best matches for the test face
        closest_distances = self.knn_clf.kneighbors(faces_encodings, n_neighbors=1)
        are_matches = [closest_distances[0][i][0] <= distance_threshold for i in range(len(X_face_locations))]
        # return all face_encodeing, face_location
        predictions  = [(pred, loc) if rec else ("unknown", loc) for pred, loc, rec in zip(self.knn_clf.predict(faces_encodings), X_face_locations, are_matches)]
        return predictions

    def run(self, frame): 
        frame = cv2.flip(frame, 1)
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        # small_frame = frame.copy()
        rgb_small_frame = small_frame[:, :, ::-1]
        # send video frame to find face locations
        X_face_locations = self.detect_face(rgb_small_frame, self.faceNet)       
        # If no faces are found in the image, return an empty result.
        if len(X_face_locations) == 0:
            cv2.line(frame, (900,0), (900,720), (255, 0, 255), 1) 
            cv2.line(frame, (900+200,0), (900+200,720), (255, 0, 255), 1) 
            fps = 1.0 / (time.time() - self.fps_time)
            self.fps_time = time.time()
            cv2.putText(frame , "FPS: %f" % (fps), (20, 40),  cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            return frame, (False, None, None)

        # Find encodings for faces in video frame
        else:
            predictions = self.face_rec(X_face_locations, rgb_small_frame)       
            this_frame_name = []
            cpy_frame = frame.copy()
            fps = 1.0 / (time.time() - self.fps_time)
            self.fps_time = time.time()
            cv2.putText(frame , "FPS: %f" % (fps), (20, 40),  cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            for name, (left, top, right, bottom) in predictions:
                # resize to original size // (1/4)*4 = 1
                top *= 4
                right *= 4 
                bottom *= 4
                left *= 4

                # draw puesdo 
                cv2.line(frame, (900,0), (900,720), (255, 0, 255), 1) 
                cv2.line(frame, (900+200,0), (900+200,720), (255, 0, 255), 1) 
                
                if not self.passFlag and right >= 900:
                    self.passFlag = True
                    self.flag = True

                if self.passFlag and right >= 900+200:
                    self.passFlag = False
            
                # draw rectangle around face
               	cv2.rectangle(frame, (left, bottom), (right, top), (0, 0, 255), 1)

                temp = self.get_temperature()
                cv2.putText(frame , "%.1f" % (temp), (left,top),  cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                # draw field for text
                cv2.rectangle(frame, (left, bottom+12), (right, bottom), (0, 0, 255), cv2.FILLED)
                
                # text setting
                font = cv2.FONT_HERSHEY_PLAIN
                cv2.putText(frame, name, (left, bottom+11), font, fontScale=1, color=(255, 255, 255), thickness=1,bottomLeftOrigin=False)
                
                if self.passFlag and self.flag: 
                    send_frame = cpy_frame[top:bottom, left:right]
                    location = (top, right, bottom, left)
                   
                    color = (0, 255, 255)
                    if temp > 37.5:
                        color = (0, 0, 255)
                    cv2.putText(send_frame,"%.1f" % (temp), (10, 35), font, fontScale=3, color=color, thickness=2,bottomLeftOrigin=False)
                    self.flag = False
                    
                    thr = Thread(target=self.send_info_to_database, args=[name, temp])
                    thr.deamon = True
                    thr.start()

                    if temp < 37.5:
                        path = os.path.join('Storage/Normal/'+str(current_milli_time()))
                        cv2.imwrite(path+'.jpg', send_frame)
                        return frame, (True, send_frame, temp)

                    elif temp >= 37.5:
                        path = os.path.join('Storage/Fever/'+str(current_milli_time()))
                        cv2.imwrite(path+'.jpg', send_frame)
                        return frame, (True, send_frame, temp)

                    
            return frame, (False, None, None)
