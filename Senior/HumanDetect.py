# A Calibrated camera that detect person, aruco marker and measure distance between those two
import cv2
import jetson_inference
import jetson_utils
import numpy as np
from cv2 import aruco
from jetson_inference import detectNet
from jetson_utils import cudaFromNumpy
import time
from Mat import rvec2rpy, homoTransMat, distance

number = 1
cap = cv2.VideoCapture(0)
time.sleep(2)
green = 200
yellow = 150
red = 100

# Load module to detect human
net = detectNet("ssd-mobilenet-v2", threshold = 0.5)
# Load Camera matrix and distortion coeeficient 
CamMat = np.load('cameramatrix.npy')
Distort = np.load('distortion.npy')
#Load ArUco dictionary, detector parameter and Aruco size in cm
Aruco_dict = cv2.aruco.getPredefinedDictionary( aruco.DICT_7X7_50)
Aruco_para = cv2.aruco.DetectorParameters_create()

A_cm = 6.8
fx = CamMat[0][0]
fy = CamMat[1][1]   # in pixel

CACHED_PTS = None
CACHED_ID = None
Line_Pts = None
measure = None
H_xmin = 0
H_xmax = 0
H_ymax = 0
H_ymin = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    #                       Undistort camera with camera matrix and distortion coeficient
    (w,h) = frame.shape[:2]
    NewCamMat, roi = cv2.getOptimalNewCameraMatrix(CamMat, Distort, (w, h), 0, (w,h))
    frame = cv2.undistort(frame, CamMat, Distort, None, NewCamMat)


    #                       Human Detection
    # Convert frame from numpy tp Cuda (transfer from CPU memory to GPU memory)

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
        

        #Find min and max values of bounding boxes


        H_xmin = int(H_topLPoint[0])
        H_xmax = int(H_topRPoint[0])
        H_ymax = int(H_botLPoint[1])
        H_ymin = int(H_topLPoint[1])

#                       Detect ArUco marker
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
            rvec, tvec, _o= cv2.aruco.estimatePoseSingleMarkers(corners[i], A_cm, CamMat, Distort)
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)
        for (markerCorner, markerID) in zip(corners, ids):

            for i in range(0, len(ids)):
                cv2.aruco.drawDetectedMarkers(frame, corners, ids)
            corners_abcd = markerCorner.reshape((4,2))
            (A_topL, A_topR, A_botR, A_botL) = corners_abcd
            A_topLPoint = (int(A_topL[0]), int(A_topL[1]))
            A_topRPoint = (int(A_topR[0]), int(A_topR[1]))
            A_botLPoint = (int(A_botL[0]), int(A_botL[1]))
            A_botRPoint = (int(A_botR[0]), int(A_botR[1]))

            A_xmin = int(A_topLPoint[0])
            A_xmax = int(A_topRPoint[0])
            A_ymax = int(A_botLPoint[1])
            A_ymin = int(A_topLPoint[1])

            # measure = A_cm/(A_ymax - A_ymin)
            # ay_px = A_topL[1] - A_botL[1]
            # ax_px = A_topL[0] - A_topR[0]
            # height_px = H_ymax - H_ymin

        #                 Assume ArUco 20 indicate Robot
            if ids[i]==20: 
                tvec_R = tvec[0]
                rvec_R = rvec[0]
                roll_R, pitch_R, yaw_R = rvec2rpy(rvec_R)
                Cam_T_Robot = homoTransMat(rvec_R, tvec_R)


        #                 ArUco 15 sticks with worker's helmet
            if ids[i] == 15:
                tvec_H = tvec[0]
                rvec_H = rvec[0]
                roll_H, pitch_H, yaw_H = rvec2rpy(rvec_H)
                Cam_T_Human = homoTransMat(rvec_H, tvec_H)

        #                 Human with ArUco tag detection
            if A_xmin > H_xmin and A_xmax < H_xmax and A_ymin > H_ymin and A_ymax < H_ymax:
                # Human height in pixel
                H_height_px = H_ymax - H_ymin
                distance_y = A_cm * fy/ ay_px
                distance_x = A_cm * fx/ ax_px
                distance = abs(distance_x)
                cv2.putText(frame, str(f"{distance:.3f}") + "cm", (int(300), int(300)), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255))

                if distance_y >= int(green):
                    cv2.putText(frame, "Green mode", (300, 100), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0))
                elif int(yellow) <= distance_y < int(green):
                    cv2.putText(frame, "Slow down", (300, 100), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 255))
                elif distance_y < int(yellow):
                    cv2.putText(frame, "STOP", (300, 100), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255))
                    print('Detected human bringing ArUco')
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
    # Quit the program when user hit 'q'
    if (key & 0XFF) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

