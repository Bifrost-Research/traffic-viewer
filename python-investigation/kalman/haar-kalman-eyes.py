# -*- coding: utf-8 -*-
"""
Created on Wed Jan 11 11:32:07 2017

@author: ragondyn
"""


import numpy as np
import cv2
execfile('tools.py')
#import tools

# Global Variables definition
#############################

MIN_H_BLUE = 100
MAX_H_BLUE = 300 
maxt = 10

stateSize = 6; #[x,y,vx,vy,w,h]
measSize = 4;#[x,y,w,h]
contrSize = 0; 

# Haar & Kalman Instantiation
#############################

face_cascade = cv2.CascadeClassifier('../../data-set/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('../../data-set/haarcascade_eye.xml')

kf = kalman_init(stateSize, measSize, contrSize)

# loop init
############
video_capture = cv2.VideoCapture(0)
ret = True;
ticks = 0
found = False; 

while True:
    ret, frame = video_capture.read()    
    res = frame.copy()
    
    precTick = ticks
    ticks = cv2.getTickCount()
    dT = (ticks - precTick)/cv2.getTickFrequency()

##########################################################################
    if(found):
        kf.transitionMatrix[2/stateSize,2%stateSize] = dT;
        kf.transitionMatrix[9/stateSize, 9%stateSize] = dT;    
        state = kf.predict()
        #print(state)
        print_Box(res, state.transpose(), (255,0,0))       
    
    ballsBox = detect_eyes(frame,res,face_cascade,eye_cascade)#detect_faces(frame,res,face_cascade)#, eye_cascade)
    if (found):
        meas_state, new = meas_state_fill(ballsBox, state)       
        state_meas_fill(state_meas, meas_state,state)
        meas = meas_format(ballsBox, state_meas, meas_state,kf)
        kf.correct(meas);  
        postcorrect = kf.statePost.transpose()
        #print(postcorrect)
        print_Box(res, postcorrect, (0,0,255))   
        #print(meas)
        #print(kf.statePost)
        
        state_meas = update_state(kf,state,state_meas, ballsBox, new)

        if np.shape(kf.statePost)[1] == 0:
            found = False
    else: 
        if (len(ballsBox) >= 1):
            found = True
            state = kalman_reset(kf,ballsBox)
            state_meas = np.zeros(np.shape(state)[1])
            
    ########################################################################       
    if(ret):    
        cv2.imshow('Video', res)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    

video_capture.release()
cv2.destroyAllWindows()