# FINAL SENIOR_
import cv2
import jetson_inference
import jetson_utils
import numpy as np
from cv2 import aruco
from jetson_inference import detectNet
from jetson_utils import cudaFromNumpy
import time
from Mat import rvec2rpy, homoTransMat, distance, relpos

# Load some initial parameters
CamMat = np.load('cameramatrix.npy')
Distort = np.load('distortion.npy')

Aruco_dict = cv2.aruco.getPredefinedDictionary( aruco.DICT_7X7_50)
Aruco_para = cv2.aruco.DetectorParameters_create()

A_length = 15 #cm
net = detectNet("ssd-mobilenet-v2", threshold = 0.5)

CACHED_PTS = None
CACHED_ID = None
Line_Pts = None
measure = None
H_xmin = 0
H_xmax = 0
H_ymax = 0
H_ymin = 0
number = 1
Cam_0 = np.zeros((1,3))
rvec, tvec = None, None
rvec_H, tvec_H = None, None
rvec_R, tvec_R = None, None

green = 150
yellow = 100


cap = cv2.VideoCapture(0)
time.sleep(2)
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

#               UNDISTORT CAMERA
    (w, h) = frame.shape[:2]
    NewCamMat, roi = cv2.getOptimalNewCameraMatrix(CamMat, Distort, (w, h), 0, (w,h))
    frame = cv2.undistort(frame, CamMat, Distort, None, NewCamMat)
#               Draw warning signs
    cv2.circle(frame, (50, 50), radius=15, color=(0, 200, 0), thickness=2 ) #green
    cv2.circle(frame, (50, 100), radius=15, color=(0, 200, 200), thickness=2 )    #yellow
    cv2.circle(frame, (50, 150), radius=15, color=(0, 0, 200), thickness=2 )    #red


#               Human Detection
    cuda_img = cudaFromNumpy(frame)
    detections = net.Detect(cuda_img, frame.shape[1], frame.shape[0])
    for detection in detections:
        class_name = net.GetClassDesc(detection.ClassID)
        conf = detection.Confidence
        H_left, H_top, H_right, H_bot = int(detection.Left), int(detection.Top), int(detection.Right), int(detection.Bottom)
    
        # Draw bounding box + Display label
        cv2.rectangle(frame, (H_left, H_top), (H_right, H_bot), (0, 255, 0), 2)
        label = f"{class_name}: {conf: .2f}"
        cv2.putText(frame, label, (H_left, H_top -5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        #Define 4 corner points of bounding box:
        H_topLPoint = (H_left, H_top)
        H_topRPoint = (H_right, H_top)
        H_botLPoint = (H_left, H_bot)
        H_botRPoint = (H_right, H_bot)

#               ArUco detection
    corners, ids, _ = cv2.aruco.detectMarkers(frame, Aruco_dict, parameters = Aruco_para)
    if len(corners) <= 0:
        if CACHED_PTS is not None:
            corners = CACHED_PTS
    if len(corners) >0:
        CACHED_PTS = corners
        if ids is not None:
            ids = ids.flatten()
            CACHED_ID = ids
        else:
            if CACHED_ID is not None:
                ids = CACHED_ID
        if len(corners) <2:
            if len(CACHED_ID) >=2:
                corners =CACHED_PTS

        for i in range(0, len(ids)):
            rvec, tvec, _o= cv2.aruco.estimatePoseSingleMarkers(corners[i], A_length, CamMat, Distort)
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)
        for (markerCorner, markerID) in zip(corners, ids):

            # for i in range(0, len(ids)):
            #     cv2.aruco.drawDetectedMarkers(frame, corners, ids)
            corners_abcd = markerCorner.reshape((4,2))
            (A_topL, A_topR, A_botR, A_botL) = corners_abcd
            A_topLPoint = (int(A_topL[0]), int(A_topL[1]))
            A_topRPoint = (int(A_topR[0]), int(A_topR[1]))
            A_botLPoint = (int(A_botL[0]), int(A_botL[1]))
            A_botRPoint = (int(A_botR[0]), int(A_botR[1]))

#           Assume ArUco 20 indicate Robot's pose
        if ids[i] ==20:
            tvec_R = tvec[0]
            rvec_R = rvec[0]
            roll_R, pitch_R, yaw_R = rvec2rpy(rvec_R)
            Cam_T_Robot = homoTransMat(rvec_R, tvec_R)


#           ArUco 15 sticks in worker's helmet
        if ids[i] ==15:
            tvec_H = tvec[0]
            rvec_H = rvec[0]
            roll_H, pitch_H, yaw_H = rvec2rpy(rvec_H)
            Cam_T_Human = homoTransMat(rvec_H, tvec_H)


#           Distance from human to robot
        if rvec_H is not None and tvec_H is not None and rvec_R is not None and tvec_R is not None:
            distance_CR = distance(tvec_R, Cam_0)
            distance_CH = distance(tvec_H, Cam_0)
            distance_RH = distance(tvec_R, tvec_H)
            cv2.putText(frame, f"Distance : {round(distance_RH ,3)}", (int(300), int(50)) , cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0))
            
            # Calculate the homogeneous transformation matrix of human wrt robot 
            Robot_T_Human, rvec_RTH, tvec_RTH = relpos(Cam_T_Robot, Cam_T_Human)
            # cv2.drawFrameAxes(frame, CamMat, Distort,rvec_RTH, tvec_RTH, 3)
            

            

            if distance_RH >= int(green):
                cv2.putText(frame, "Safe", (100, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0))
                cv2.circle(frame, (50, 50), radius=15, color=(0, 200, 0), thickness=cv2.FILLED ) #green

            elif int(yellow) <= distance_RH < int(green):
                cv2.putText(frame, "Slow down", (100, 100), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 255))
                cv2.circle(frame, (50, 100), radius=15, color=(0, 200, 200), thickness=cv2.FILLED )    #yellow
            elif distance_RH < int(yellow):
                cv2.putText(frame, "STOP", (100, 150), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255))
                cv2.circle(frame, (50, 150), radius=15, color=(0, 0, 255), thickness=cv2.FILLED )    #RED   
            else:
                continue

    cv2.imshow("Detection", frame)
    key = cv2.waitKey(1)        
# Capture image when user hit 'c'
    if (key & 0XFF) == ord('c'):
        if (number <10):
            fname = 'Result0' + str(number) + '.jpg'
        else:
            fname = 'Result' + str(number) + '.jpg'
        cv2.imwrite(fname, frame)
        number = number +1
        print("Pose of Human wrt Robot: \n", Robot_T_Human)
        print("Pose of Human wrt Camera: \n", Cam_T_Human)
        print("Pose of Robot wrt Camera: \n", Cam_T_Robot)

    # Quit the program when user hit 'q'
    if (key & 0XFF) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()