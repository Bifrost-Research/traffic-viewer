# -*- coding: utf-8 -*-
"""
Created on Tue Dec 20 21:47:20 2016

@author: ragondyn
"""

import numpy as np
import cv2
import time

img = cv2.imread('../pictures/fgmaskmog.png')
cv2.imshow('mask',img)

kernel = np.ones((10,10),np.uint8)

closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

cv2.imshow('mask-closed',closing)

if cv2.waitKey(0) & 0xFF == ord('q'):
    cv2.destroyAllWindows()