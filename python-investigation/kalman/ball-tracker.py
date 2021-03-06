# -*- coding: utf-8 -*-
"""
Created on Wed Dec 21 00:38:29 2016

@author: ragondyn
"""

import numpy as np
import cv2
# Kalman initialisation

MIN_H_BLUE = 200
MAX_H_BLUE = 300

stateSize = 6; #[x,y,vx,vy,w,h]
measSize = 4;#[x,y,w,h]
contrSize = 0; #control vector?

kf = cv2.KalmanFilter(stateSize, measSize, contrSize);

state = np.zeros(stateSize, dtype = np.float32)
meas = np.zeros(measSize, dtype = np.float32)

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

video_capture = cv2.VideoCapture(0)
ret = True;

ticks = 0
found = False;
notFoundCount = 0;  

while True:
    ret, frame = video_capture.read()    
    res = frame.copy()
    
    precTick = ticks
    ticks = cv2.getTickCount()
    dT = (ticks - precTick)/cv2.getTickFrequency()
    
    ## Prediction part
    
    # if previous iteration it in "found" mode (ie we detected it, or 
    # it was detected not so long ago, it must still be here)
    if(found):
        # define the transormation the object is experiencing
        kf.transitionMatrix[2/stateSize,2%stateSize] = dT;
        kf.transitionMatrix[9/stateSize, 9%stateSize] = dT;
        
        # predict future state
        state = kf.predict()
        
        # print in blue predicted state
        width = state[4];          
        height = state[5];          
        x = state[0] - width / 2;          
        y = state[1] - height / 2;            
        cx = state[0];          
        cy = state[1];          
        cv2.circle(res, (cx,cy), 2, (255,0,0), -1);            
        cv2.rectangle(res, (x,y), (x+width,y+height), (255,0,0), 2); #
    
    # II Detection part
    
    # detection of the position of the object thanks to color
    # based detection
    
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
        if (ratio > 0.75 and w*h >= 400):
            balls.append(contours[1][i]);
            ballsBox.append(bBox);
    
    # Filtering
    for i in range(0,len(balls)):
        x,y,h,w = ballsBox[i]
        cv2.drawContours(res, balls, i, (20,150,20), 1);
        cv2.rectangle(res, (x,y), (x+w,y+h), (0,255,0));
        cx = x + w/ 2;
        cy = y + h/ 2;
        cv2.circle(res, (cx,cy), 2, (20,150,20), -1);
        
    # If object not detected, increment notcount (will evaluate if the 
    #    prediction is still relevent)
        
    if (len(balls) == 0):
        notFoundCount += 1       
        if( notFoundCount >= 10 ):
            # too old, give up
            found = False;
        else:
            # still relevent, but no observation detected (the observation is 
            # therefore the prediction, no error)
            kf.statePost = state;
    else: 
        # mesurements of the detected object
        notFoundCount = 0;
        x,y,h,w = ballsBox[0]
        meas[0] = x + w/ 2;
        meas[1] = y + h/ 2;
        meas[2] = float(w);
        meas[3] = float(h);
        
        if not found: # first detection, the object just has been found,
            # we need to initialize its first state
            # the error cov is Id matrix
            kf.errorCovPre[0,0] = 1; # px
            kf.errorCovPre[7/6,7%6] = 1; # px
            kf.errorCovPre[14/6,14%6] = 1;
            kf.errorCovPre[21/6,21%6] = 1;
            kf.errorCovPre[28/6,28%6] = 1; # px
            kf.errorCovPre[35/6,35%6] = 1; # px
            
            # velocity null, and state = measures
            state[0] = meas[0];
            state[1] = meas[1];
            state[2] = 0;
            state[3] = 0;
            state[4] = meas[2];
            state[5] = meas[3];            
            
            # no error to measure, because there isn't anything to predict yet
            kf.statePost = state;
 
            found = True;
        else:
            # clssic scenario, correct the model thanks to measurements
            kf.correct(meas);
            print("error")
            print(kf.errorCovPre)
    if(ret):    
        cv2.imshow('Video', res)
#        cv2.waitKey(int(np.floor(1/fps*1000*1/alpha)))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    

video_capture.release()
cv2.destroyAllWindows()