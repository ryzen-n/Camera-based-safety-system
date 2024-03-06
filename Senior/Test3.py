import cv2
from cv2 import aruco
import numpy as np
import math

aruco_dict = cv2.aruco.getPredefinedDictionary(aruco.DICT_7X7_100)
aruco_para = cv2.aruco.DetectorParameters_create()
CamMat = np.load('cameramatrix.npy')
Distort = np.load('distortion.npy')
img = cv2.imread('./Pose_Img03.jpg')
length = 6.8 #cm

corner, id, _= cv2.aruco.detectMarkers(img, aruco_dict, parameters = aruco_para)
if id is not None and len(id) >=2:
    for i in range(0, len(id)):

        aruco.drawDetectedMarkers(img, corner)
        marker_robot = 20
        marker_human = 15

        rvec, tvec, _o= cv2.aruco.estimatePoseSingleMarkers(corner[i], length, CamMat, Distort)
        if id[i] == 20:  
            tvec_R = tvec[0]
            rvec_R = rvec[0]
        if id[i] == 15:
            tvec_H = tvec[0]
            rvec_H = rvec[0]
        
        cv2.drawFrameAxes(img, CamMat, Distort, rvec_H, tvec_H, 3)

cv2.imshow("Result", img)
cv2.waitKey(0)
cv2.destroyAllWindows()