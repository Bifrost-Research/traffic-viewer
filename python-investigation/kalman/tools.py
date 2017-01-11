# -*- coding: utf-8 -*-
"""
Created on Sun Dec 25 21:31:18 2016

@author: ragondyn
"""

import numpy as np
import cv2

MIN_H_BLUE = 100
MAX_H_BLUE = 300 
maxt = 10

stateSize = 6; #[x,y,vx,vy,w,h]
measSize = 4;#[x,y,w,h]
contrSize = 0; 

def kalman_init(stateSize, measSize, contrSize):
    kf = cv2.KalmanFilter(stateSize, measSize, contrSize);
    #   // Transition State Matrix A
    #   // Note: set dT at each processing step!
    #   // [ 1 0 dT 0  0 0 ]
    #   // [ 0 1 0  dT 0 0 ]
    #   // [ 0 0 1  0  0 0 ]
    #   // [ 0 0 0  1  0 0 ]
    #   // [ 0 0 0  0  1 0 ]
    #   // [ 0 0 0  0  0 1 ]
    kf.transitionMatrix = np.array(np.diag(np.repeat(1,stateSize)),dtype=np.float32)
    
    #   // Measure Matrix H
    #   // [ 1 0 0 0 0 0 ]
    #   // [ 0 1 0 0 0 0 ]
    #   // [ 0 0 0 0 1 0 ]
    #   // [ 0 0 0 0 0 1 ]
    kf.measurementMatrix = np.zeros((measSize,stateSize),dtype = np.float32)
    kf.measurementMatrix[0,0] = 1.0;
    kf.measurementMatrix[7/stateSize, 7%stateSize] = 1.0;
    kf.measurementMatrix[16/stateSize, 16%stateSize] = 1.0;
    kf.measurementMatrix[23/stateSize , 23%stateSize] = 1.0;
    
    #   // Process Noise Covariance Matrix Q
    #   // [ Ex 0  0    0   0    0 ]
    #   // [ 0  Ey 0    0   0    0 ]
    #   // [ 0  0  Ev_x 0   0    0 ]
    #   // [ 0  0  0   Ev_y 0    0 ]
    #   // [ 0  0  0    0   Ew   0 ]
    #   // [ 0  0  0    0   0    Eh]
    kf.processNoiseCov = kf.processNoiseCov * 1e-2
    kf.processNoiseCov[0,0] = 1e-2;
    kf.processNoiseCov[7/stateSize,7%stateSize] = 1e-2;
    kf.processNoiseCov[14/stateSize,14%stateSize] = 2.0;
    kf.processNoiseCov[21/stateSize,21%stateSize] = 1.0;
    kf.processNoiseCov[28/stateSize,28%stateSize] = 1e-2;
    kf.processNoiseCov[35/stateSize,35%stateSize] = 1e-2;
    
    
    kf.measurementNoiseCov = kf.measurementNoiseCov * 1e-1
    return kf

def print_Box(frame, Box_list, col):
    for i in range(0,len(Box_list)):
        width = int(Box_list[i][4]);          
        height = int(Box_list[i][5]);          
        x = int(Box_list[i][0]) #- width / 2);          
        y = int(Box_list[i][1]) #- height / 2);                     
        cv2.rectangle(frame, (x,y), (x+width,y+height), col, 2); #

def detect_ball(frame,res):
    # Gaussian smoothing
    blur = cv2.GaussianBlur(frame,(5,5),3.0,3.0) 
    
    # HSV conversion
    frmHsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    
    # Thresholding
    rangeRes = np.zeros(frame.size,dtype = 'uint8')
    rangeRes = cv2.inRange(frmHsv, (MIN_H_BLUE / 2, 100, 80),
            (MAX_H_BLUE / 2, 255, 255))
    
    rangeRes = cv2.erode(rangeRes, kernel = np.ones((3,3),dtype = 'uint8'), 
                         anchor = (-1, -1), iterations = 2);
    rangeRes = cv2.dilate(rangeRes, np.ones((3,3), dtype = 'uint8'),
                          anchor = (-1, -1), iterations = 2);
    # Thresholding viewing  
    cv2.imshow("Threshold", rangeRes);         
    # Contours detection
    contours = cv2.findContours(rangeRes, mode = cv2.RETR_EXTERNAL,
            method = cv2.CHAIN_APPROX_NONE);
        
    balls = [];
    ballsBox = [];
    
    for i in range(0,len(contours[1])):
        bBox = cv2.boundingRect(contours[1][i]);   
        x,y,w,h = bBox        
        ratio = float(w) / float(h);  
        if (ratio > 1.0):
            ratio = 1.0 / ratio;
         # Searching for a bBox almost square
        if (ratio > 0.8 and w*h >= 400):
            balls.append(contours[1][i]);
            ballsBox.append(bBox);
    
    # Filtering and drawing
    for i in range(0,len(balls)):
        x,y,w,h = ballsBox[i]
        cv2.drawContours(res, balls, i, (20,150,20), 1);
        cv2.rectangle(res, (x,y), (x+w,y+h), (0,255,0));
        cx = x #+ w/ 2;
        cy = y #+ h/ 2;
        cv2.circle(res, (cx,cy), 2, (20,150,20), -1);
    return((ballsBox))

def detect_faces(frame,res,face_cascade):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces: 
        cv2.rectangle(res, (x,y), (x+w,y+h), (0,255,0));
    return(faces)

def detect_eyes(frame,res,face_cascade,eye_cascade):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    eyes = []
    for (x,y,w,h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        eyes.extend(np.array((x,y,0,0)) +
            (eye_cascade.detectMultiScale(roi_gray)))
    for (x,y,w,h) in eyes:        
        cv2.rectangle(res, (x,y), (x+w,y+h), (0,255,0));
    return(eyes)

def kalman_reset(kf,ballsBox):
    kf.errorCovPre[0,0] = 1; # px
    kf.errorCovPre[7/6,7%6] = 1; # px
    kf.errorCovPre[14/6,14%6] = 1;
    kf.errorCovPre[21/6,21%6] = 1;
    kf.errorCovPre[28/6,28%6] = 1; # px
    kf.errorCovPre[35/6,35%6] = 1; # px
    
    # velocity null, and state = measures
    state = np.zeros((stateSize,len(ballsBox)),dtype = np.float32)
    for i in range(0,len(ballsBox)):
        x,y,w,h = ballsBox[i]
        state[0][i] = x #+ w/ 2;
        state[1][i] = y #+ h/2;
        state[2][i] = 0;
        state[3][i] = 0;
        state[4][i] = w;
        state[5][i] = h;            
    
    # no error to measure, because there isn't anything to predict yet
    kf.statePost = state;
    return state
     
# match each observation with a states index
# each state is assigned to maximum ONE observation 
def meas_state_fill(ballsBox, state):
    meas_state = np.ones(len(ballsBox)) * (-1)
    new = []
    norms = np.zeros((len(ballsBox),np.shape(state)[1]))
    for i in range(0,len(ballsBox)):
        x,y,w,h = ballsBox[i]
        for j in range(0,np.shape(state)[1]):
            norms[i,j] = np.linalg.norm(np.array([x,y]) - state[[0,1] ,j])
    
    for i in range(0,len(ballsBox)):
        meas_state[i] =  list(norms[i,:]).index(min(norms[i,:]))
        index = [l for l,x in enumerate(meas_state) if x == meas_state[i]]      
        curr = i   
        back = range(0,np.shape(state)[1])
        while len(index) > 1:
            if norms[index[0],meas_state[curr]] > norms[index[1],meas_state[curr]]:
                furthest = index[0]
                closest = index[1]
            else:
                furthest = index[1]
                closest = index[0]
            
            back.remove(meas_state[closest])
            if len(back)>0:
                meas_state[furthest] = list(norms[furthest,:]).index(min(norms[furthest,back]))
                curr = furthest
                index = [i for i,x in enumerate(meas_state) if x == meas_state[curr]]
            else:
                meas_state[furthest] = -1
                index = []
    
    new = [l for l,x in enumerate(meas_state) if x == -1]
    return meas_state, new
            
            
    for k in range(0,np.shape(state)[1]):
        for j in range(0,np.shape(state)[1]):
            index = range(0,len(ballsBox))
            b = True
            while b:
                if(len(index)>0):
                    i = list(norms[:,j]).index(min(norms[index,j]))
                    if norms[i,j] == min(norms[i,range(j,np.shape(state)[1])]):
                        b = False
                        meas_state[i] = j
                    else:
                        index.remove(i)
                else:
                    b = False
        
    return meas_state,new
    
def state_meas_fill(state_meas, meas_state,state):
    state_not_detected = list(np.linspace(0,np.shape(state)[1]-1, np.shape(state)[1]))
    for i in range(0,len(meas_state)):
        if meas_state[i]>=0:
            state_meas[int(meas_state[i])] = i
            if meas_state[i] in state_not_detected:
                state_not_detected.remove(meas_state[i])
    for i in state_not_detected:
        # new state not detected
        if(state_meas[int(i)] >= 0):
            state_meas[int(i)] = -1
        state_meas[int(i)] -= 1
        
def meas_format(ballsBox, state_meas, meas_state,kf):
    # find related measures for each old states (detected or not)
    meas = np.zeros((measSize,len(state_meas)),dtype = np.float32)   
    for i in range(0,len(state_meas)):
        if state_meas[i] >= 0:
            x,y,w,h = ballsBox[int(state_meas[i])]
            # selecting state's index object for given measures
            meas[0][i] = x # + w/ 2;
            meas[1][i] = y # + h/ 2;
            meas[2][i] = float(w);
            meas[3][i] = float(h);
            
        else:
            x,y,w,h = kf.statePre[[0,1,4,5],i]
            meas[0][i] = x #+ w/ 2;
            meas[1][i] = y #+ h/ 2;
            meas[2][i] = float(w);
            meas[3][i] = float(h);    
    return meas
            

def update_state(kf,state,state_meas, ballsBox, new):
    old = []
    for i in range(0,len(state_meas)):
        if (state_meas[i] < 0):
            kf.statePost[:,i] = state[:,i]
    
    for i in range(0,len(state_meas)):
        if (state_meas[i] > -maxt):
            old.append(i)
    kf.statePost = kf.statePost[:,old]
    state_meas = state_meas[old]
    #and add
    new_states = np.zeros((stateSize, len(new)), dtype = np.float32)
    k = 0        
    for i in new:
        x,y,w,h = ballsBox[i]
        new_states[0][k] = x #+ w/ 2;
        new_states[1][k] = y #+ h/ 2;
        new_states[4][k] = float(w);
        new_states[5][k] = float(h);
        k += 1
    kf.statePost = np.concatenate((kf.statePost,new_states),1) 
    state_meas = np.concatenate((state_meas,np.zeros(len(new))))
    return(state_meas)