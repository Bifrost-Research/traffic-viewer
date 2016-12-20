# -*- coding: utf-8 -*-
"""
Created on Mon Dec 19 18:32:13 2016

@author: ragondyn
"""
import numpy as np
import cv2
import time

video_capture = cv2.VideoCapture(
    #"http://98.102.110.114:82/mjpg/video.mjpg?timestamp=1482240533845")
    "http://206.176.34.51/mjpg/video.mjpg")

k = 2*2+1 # window size
ret = True;

# initialising MOG model
fgbg_mog = cv2.bgsegm.createBackgroundSubtractorMOG(backgroundRatio=0.5, 
                                                    nmixtures=20)
# the more is the ratio, the less we see, the more nmixture is, the more 
# accurate the mask is

# initialising MOG2 model -> noise, add cv filter
fgbg_mog2 = cv2.createBackgroundSubtractorMOG2(detectShadows = False,varThreshold=70)
#if varThreshold high, similar to default mog 
# initialising GMG model
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
fgbg_gmg = cv2.bgsegm.createBackgroundSubtractorGMG()


while ret:
    # Capture frame-by-frame
    ret, frame = video_capture.read()    
    #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if(ret):    
        cv2.imshow('Video', frame)
        t0 = time.time()
        
        # MOG update
        fgmask_mog = fgbg_mog.apply(frame) 
        fgmask_mog = cv2.medianBlur(fgmask_mog,k)        
        cv2.imshow('Foreground mask MOG', fgmask_mog)
        tmog = time.time() - t0
        
        # MOG2 update
        fgmask_mog2 = fgbg_mog2.apply(frame)
        fgmask_mog2 = cv2.medianBlur(fgmask_mog2,k)
        cv2.imshow('Foreground mask MOG2', fgmask_mog2)
        tmog2 = time.time() - tmog    
        
        # GMG update
        fgmask_gmg = fgbg_gmg.apply(frame)
        fgmask_gmg = cv2.morphologyEx(fgmask_gmg, cv2.MORPH_OPEN, kernel)
        fgmask_gmg = cv2.medianBlur(fgmask_gmg,k)
        cv2.imshow('Foreground mask GMG', fgmask_gmg)
        tgmg = time.time() - tmog2
        
        print('tmog:', tmog, 'tmog2:', tmog2, 'tgmg :', tgmg)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()