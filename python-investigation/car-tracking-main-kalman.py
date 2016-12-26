# -*- coding: utf-8 -*-
"""
Created on Sun Dec 25 21:40:56 2016

@author: ragondyn
"""
import numpy as np
import cv2
import time
execfile('kalman/tools.py')

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
    #"../data-set/kiwanis-day.mjpg")
    0)

# only if recorded - not live
alpha = 1 # spead-up the video
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



# kalman
kf = kalman_init(6, 4, 0)

ticks = 0
found = False; 

while ret:
    # Capture frame-by-frame
    ret, frame = video_capture.read()    
    if ret :
        frame2 = frame.copy()
            
        # Time update
        t0 = time.time()
        precTick = ticks
        ticks = cv2.getTickCount()
        dT = (ticks - precTick)/cv2.getTickFrequency()
        
        # Kalman prediction and update transition
        ########################################
        
        if(found):
            kf.transitionMatrix[2/stateSize,2%stateSize] = dT;
            kf.transitionMatrix[9/stateSize, 9%stateSize] = dT;    
            state = kf.predict()
            print_Box(frame2, state[[0,1,4,5],:].transpose(), (255,0,0))         
        
        # Detection part
        ################
        
        # MOG update
        fgmask_mog = fgbg_mog.apply(frame) 
        fgmask_mog = cv2.medianBlur(fgmask_mog,k)        
    
        # Morphological tf
        fgmask_mog = cv2.morphologyEx(fgmask_mog, cv2.MORPH_CLOSE, kernel)
        
        # Extract blob positions
        res = extract_pos(detector,fgmask_mog)
        
        fgmask_mog_kpts = cv2.drawKeypoints(fgmask_mog, res, np.array([]), 
                               (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        cv2.imshow('Foreground mask MOG and detected cars', fgmask_mog_kpts)
        # extract info to fit kalman format
        ballsBox = []        
        for i in res:
            x,y = i.pt
            size = i.size
            x -= size/2
            y -= size/2
            ballsBox.append((int(x), int(y), int(size), int(size)))
            
            
        if (found):
            meas_state, new = tools.meas_state_fill(ballsBox, state) 
            state_meas_fill(state_meas, meas_state,state)
            meas = meas_format(ballsBox, state_meas, meas_state,kf)
            print_Box(frame2, meas.transpose(), (0,255,0))         
            kf.correct(meas);        
            
            state_meas = update_state(kf,state,state_meas, ballsBox, new)
            
            if np.shape(kf.statePost)[1] == 0:
                found = False
        else: 
            if (len(ballsBox) >= 1):
                found = True
                state = kalman_reset(kf,ballsBox)
                state_meas = np.zeros(np.shape(state)[1])

        cv2.imshow('Video', frame2)    
        if cv2.waitKey(int(np.floor(1/fps*1000*1/alpha))) & 0xFF == ord('q'):
            break

video_capture.release()
cv2.destroyAllWindows()