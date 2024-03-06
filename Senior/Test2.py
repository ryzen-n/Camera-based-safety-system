import cv2
from cv2 import aruco
import numpy as np
import math
import time
from Mat import rvec2rpy, homoTransMat, distance, relpos


cap = cv2.VideoCapture(0)
time.sleep(2)

#Aruco information:
aruco_dict = cv2.aruco.getPredefinedDictionary(aruco.DICT_7X7_100)
aruco_para = cv2.aruco.DetectorParameters_create()
CamMat = np.load('cameramatrix.npy')
Distort = np.load('distortion.npy')
length = 15 #cm

rvec, tvec = None, None
rvec_H, tvec_H = None, None
rvec_R, tvec_R = None, None
Cam_0 = np.zeros((1,3))

CACHED_PTS = None
CACHED_IDS = None
CACHED_PTS_2 = None
CACHED_IDS_2 = None

while True:
    ret, frame = cap.read()
    (w,h) = frame.shape[:2]
    NewCamMat, roi = cv2.getOptimalNewCameraMatrix(CamMat, Distort, (w, h), 0, (w,h))
    frame = cv2.undistort(frame, CamMat, Distort, None, NewCamMat)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    corner, id, _= cv2.aruco.detectMarkers(gray, aruco_dict, parameters = aruco_para)
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
        if id[i] == 15:
            tvec_H = tvec[0]
            rvec_H = rvec[0]
            roll_H, pitch_H, yaw_H = rvec2rpy(rvec_H)
            Cam_T_Human = homoTransMat(rvec_H, tvec_H)
        
        if rvec_H is not None and tvec_H is not None and rvec_R is not None and tvec_R is not None:
            distance_CR = distance(tvec_R, Cam_0)
            distance_CH = distance(tvec_H, Cam_0)
            distance_RH_1 = distance(tvec_R, tvec_H)
            cv2.putText(frame, f"Distance 1: {round(distance_RH_1 ,3)}", (int(50), int(50)) , cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 250, 0))
            
            # Calculate the homogeneous transformation matrix of human wrt robot 
            Robot_T_Human, rvec_RTH, tvec_RTH = relpos(Cam_T_Robot, Cam_T_Human)
            # cv2.drawFrameAxes(frame, CamMat, Distort,rvec_RTH, tvec_RTH, 3)
            print("Pose of Human wrt Robot", Robot_T_Human)

            distance_RH_2 = math.sqrt((tvec_RTH[0,0]**2 + tvec_RTH[0,1]**2 +tvec_RTH[0,2]**2))
            cv2.putText(frame, f"Distance 2: {round(distance_RH_2 ,3)}", (int(50), int(100)) , cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255))
    #Display result
    cv2.imshow("Result", frame)
    if cv2.waitKey(1) & 0XFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
