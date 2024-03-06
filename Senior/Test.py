import cv2
from cv2 import aruco
import numpy as np
import math
from Mat import rvec2rpy, homoTransMat, distance, relpos


#Aruco information:
aruco_dict = cv2.aruco.getPredefinedDictionary(aruco.DICT_7X7_100)
aruco_para = cv2.aruco.DetectorParameters_create()
CamMat = np.load('cameramatrix.npy')
Distort = np.load('distortion.npy')
frame = cv2.imread('./Result01.jpg')
length = 15 #cm
axis = np.float32([[-.01, -.01, 0], [-.01, .01, 0], [.01, -.01, 0], [.01, .01, 0]]).reshape(-1, 3)

rvec, tvec = None, None
rvec_H, tvec_H = None, None
rvec_R, tvec_H = None, None

CACHED_PTS = None
CACHED_IDS = None
CACHED_PTS_2 = None
CACHED_IDS_2 = None

composedRvec, composedTvec = None, None
Cam_O = np.zeros((1,3))

# (w,h) = img.shape[:2]
# NewCamMat, roi = cv2.getOptimalNewCameraMatrix(CamMat, Distort, (w, h), 0, (w,h))
# frame = cv2.undistort(img, CamMat, Distort, None, NewCamMat)

corner, id, _= cv2.aruco.detectMarkers(frame, aruco_dict, parameters = aruco_para)
if len(corner) <= 0:
        if CACHED_PTS is not None:
            corner = CACHED_PTS

if len(corner) >0:
    CACHED_PTS = corner
    if id is not None:
        id = id.flatten()
        CACHED_IDS = id
    else:
        if CACHED_IDS is not None:
            id = CACHED_IDS
    if len(corner) < 2:
        if len(CACHED_IDS) >=2:
            corner = CACHED_PTS
    for i in range(0, len(id)):
        rvec, tvec, _o= cv2.aruco.estimatePoseSingleMarkers(corner[i], length, CamMat, Distort)
        cv2.aruco.drawDetectedMarkers(frame, corner, id)
        # cv2.drawFrameAxes(frame, CamMat, Distort, rvec, tvec, 3)

        # Assume ArUco 20 indicates Robot
        if id[i] == 20:  
            tvec_R = tvec[0]
            rvec_R = rvec[0]
            roll_R, pitch_R, yaw_R = rvec2rpy(rvec_R)
            Cam_T_Robot = homoTransMat(rvec_R, tvec_R)

        # Assume ArUco 15 idicates human
        elif id[i] == 15:
            tvec_H = tvec[0]
            rvec_H = rvec[0]
            roll_H, pitch_H, yaw_H = rvec2rpy(rvec_H)
            Cam_T_Human = homoTransMat(rvec_H, tvec_H)
        
    if rvec_H is not None and tvec_H is not None and rvec_R is not None and tvec_R is not None:
        distance_CR = distance(tvec_R, Cam_O)
        distance_CH = distance(tvec_H, Cam_O)
        distance_RH = distance(tvec_H, tvec_R)

        R_T_H, rvec_RTH, tvec_RTH = relpos(Cam_T_Robot, Cam_T_Human)

        print("Homogeneous transformation of human wrt camera: \n", Cam_T_Human)
        print("Homogeneous transformation of robot wrt camera: \n", Cam_T_Robot)
        print("Homogeneous transformatrion of human wrt robot: \n", R_T_H)
        
        cv2.drawFrameAxes(frame, CamMat, Distort, rvec_RTH, tvec_RTH, 3)


        # Calculate the homogeneous transformation matrix of human wrt robot 

        # print("Pose of Human wrt Robot", Robot_T_Human)
cv2.imshow("Result", frame)

cv2.waitKey(0)
cv2.destroyAllWindows()
