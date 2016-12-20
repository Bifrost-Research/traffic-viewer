# -*- coding: utf-8 -*-
"""
Created on Mon Dec 19 18:32:13 2016

@author: ragondyn
"""
import numpy as np
import cv2
import os

video_capture = cv2.VideoCapture(
    "http://206.176.34.51/mjpg/video.mjpg")
alpha = 1 # spead-up the video
fps = (video_capture.get(cv2.CAP_PROP_FPS))

ret = True;

while ret:
    #time = (video_capture.get(cv2.CV_CAP_PROP_POS_MSEC))
    #print (time)

    # There seem to be a problem with timestamping pictures from webcam stream
    # http://answers.opencv.org/question/100052/opencv-videocapturegetcv_cap_prop_pos_msec-returns-0/
    # https://github.com/opencv/opencv/blob/master/modules/videoio/src/cap_v4l.cpp
    # that is only a problem for measuring the speed of the cars, and we can still
    # use time from the system to evaluate it (even if its not very accurate)
    
    # Capture frame-by-frame
    ret, frame = video_capture.read()    
    #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if(ret):    
        cv2.imshow('Video', frame)
#        cv2.waitKey(int(np.floor(1/fps*1000*1/alpha)))
    if cv2.waitKey(int(np.floor(1/fps*1000*1/alpha))) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
