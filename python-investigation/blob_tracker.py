# -*- coding: utf-8 -*-
"""
Created on Tue Dec 20 15:29:48 2016

@author: ragondyn
"""

import numpy as np
import cv2
import time

def extract_pos(detector, fgmask_img):
    return detector.detect(fgmask_img)
    
def init_detector():
    # Set up the detector with default parameters.
    params = cv2.SimpleBlobDetector_Params()
    params.blobColor = 255
    params.filterByColor = True
    #params.minDistBetweenBlobs = 90
    
    # Change thresholds
    params.minThreshold = 10;
    params.maxThreshold = 200;
     
    # Filter by Area.
    params.filterByArea = True
    params.minArea = 100
     
    # Filter by Circularity
    params.filterByCircularity = False
    params.minCircularity = 0.1
     
    # Filter by Convexity
    params.filterByConvexity = False
    params.minConvexity = 0.87
     
    # Filter by Inertia
    params.filterByInertia = False
    params.minInertiaRatio = 0.01
    
    detector = cv2.SimpleBlobDetector_create(params)
    return detector