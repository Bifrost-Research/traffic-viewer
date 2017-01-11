# -*- coding: utf-8 -*-
"""
Created on Mon Dec 19 18:32:13 2016

@author: ragondyn
"""
import numpy as np
import cv2
import os

# To investigate the method, we will first implement a simple face and eye
# detector with the training set given by opencv

video_capture = cv2.VideoCapture(0)
   # "http://206.176.34.51/mjpg/video.mjpg")
alpha = 1 # spead-up the video
fps = (video_capture.get(cv2.CAP_PROP_FPS))


# Init haar-cascade
face_cascade = cv2.CascadeClassifier('../../data-set/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('../../data-set/haarcascade_eye.xml')

ret = True;
while ret:
    # Capture frame-by-frame
    ret, frame = video_capture.read()    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
    
    
    if(ret):    
        cv2.imshow('Video', frame)
#        cv2.waitKey(int(np.floor(1/fps*1000*1/alpha)))
    if cv2.waitKey(int(np.floor(1/fps*1000*1/alpha))) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
