import cv2
from cv2 import aruco
import numpy as np

img = cv2.imread('./Input_Img/logi/PImg23.jpg')
CamMat = np.load('cameramatrix.npy')
Dist   = np.load('distortion.npy')


img_calib = cv2.undistort(img, CamMat, Dist, None)
cv2.imshow('Original image', img)
cv2.imshow('Undistort image', img_calib)
cv2.waitKey()
cv2.destroyAllWindows()