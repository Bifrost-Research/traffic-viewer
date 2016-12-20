# -*- coding: utf-8 -*-
"""
Created on Mon Dec 19 18:32:13 2016

@author: ragondyn
"""
import numpy as np
import cv2
import time

# import tracking function from specified file
from blob_tracker import extract_pos, init_detector

video_capture = cv2.VideoCapture(
    #"http://98.102.110.114:82/mjpg/video.mjpg?timestamp=1482240533845") # i-75
    #"http://206.176.34.51/mjpg/video.mjpg") # louise
    #"http://gieat.viewsurf.com/?id=5584&action=mediaRedirect") #grenoble
    #http://206.176.34.53/mjpg/video.mjpg?COUNTER) #sioux
    #"../data-set/louise-day.mjpg")
    #"../data-set/I-75-day.mjpg")
    #"../data-set/grenoble-night.mjpg")
    "../data-set/kiwanis-day.mjpg")
# only if recorded - not live
alpha = 0.1 # spead-up the video
fps = (video_capture.get(cv2.CAP_PROP_FPS))
if fps == 0:
    fps = 25

k = 2*2+1 # window size for smoothing in foreground segmentation
ret = True;

# using MOG foreground segmentation
# initialising MOG model
fgbg_mog = cv2.bgsegm.createBackgroundSubtractorMOG(history = 500,backgroundRatio=0.1, 
                          nmixtures=15, )
# for closure completion - morphological tf
kernel = np.ones((10,10),np.uint8)
# 20 works well for kiwanis


detector = init_detector()

while ret:
    # Capture frame-by-frame
    ret, frame = video_capture.read()    
    #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if(ret):    
        t0 = time.time()
        
        # MOG update
        fgmask_mog = fgbg_mog.apply(frame) 
        fgmask_mog = cv2.medianBlur(fgmask_mog,k)        

        # Morphological tf
        fgmask_mog = cv2.morphologyEx(fgmask_mog, cv2.MORPH_CLOSE, kernel)
        
        # Extract blob positions
        res = extract_pos(detector,fgmask_mog) 
        
        # Draw circles
        fgmask_mog_kpts = cv2.drawKeypoints(fgmask_mog, res, np.array([]), 
                               (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        cv2.imshow('Foreground mask MOG and detected cars', fgmask_mog_kpts)
        frame_kpts = cv2.drawKeypoints(frame, res, np.array([]), 
                                (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        cv2.imshow('Detected cars', frame_kpts)
        
    if cv2.waitKey(int(np.floor(1/fps*1000*1/alpha))) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()