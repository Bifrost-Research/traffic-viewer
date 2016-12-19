# -*- coding: utf-8 -*-
"""
Created on Mon Dec 19 18:32:13 2016

@author: ragondyn
"""
import numpy as np
import cv2
import os

video_capture = cv2.VideoCapture(
    "http://gieat.viewsurf.com/?id=5584&action=mediaRedirect")
fps = (video_capture.get(cv2.CAP_PROP_FPS))
alpha = 5 # spead-up the video
ret = True;

while ret:
    # Capture frame-by-frame
    ret, frame = video_capture.read()    
    #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if(ret):    
        cv2.imshow('Video', frame)
        cv2.waitKey(int(np.floor(1/fps*1000*1/alpha)))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
