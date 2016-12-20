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
    #"http://98.102.110.114:82/mjpg/video.mjpg?timestamp=1482240533845")
    "http://206.176.34.51/mjpg/video.mjpg")

k = 2*2+1 # window size for smoothing in foreground segmentation
ret = True;

# using MOG foreground segmentation
# initialising MOG model
fgbg_mog = cv2.bgsegm.createBackgroundSubtractorMOG(backgroundRatio=0.2, 
                                                    nmixtures=15)
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
        tmog = time.time() - t0
        
        res = extract_pos(detector,fgmask_mog) 
        
        # draw circles
        fgmask_mog_kpts = cv2.drawKeypoints(fgmask_mog, res, np.array([]), 
                               (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        cv2.imshow('Foreground mask MOG and detected cars', fgmask_mog_kpts)
        frame_kpts = cv2.drawKeypoints(frame, res, np.array([]), 
                                (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        cv2.imshow('Detected cars', frame_kpts)
        
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()