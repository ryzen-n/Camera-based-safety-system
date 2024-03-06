# This code find the distance between two aruco markers

import cv2
from cv2 import aruco
import numpy as np
import time
import imutils
from imutils.video import VideoStream

vs = VideoStream(src=0).start()
time.sleep(2)
aruco_dict = cv2.aruco.getPredefinedDictionary(aruco.DICT_7X7_50)
aruco_param = cv2.aruco.DetectorParameters_create()


CACHED_PTS = None
CACHED_IDS = None
Line_Pts = None
measure = None

CACHED_PTS_2 = None
CACHED_IDS_2 = None
Line_Pts_2 = None
measure_2 = None

CamMat = np.load('./cameramatrix.npy')
Distort = np.load('./distortion.npy')

while True:
    Dist =[]
    image = vs.read()
    # image = imutils.resize(image, width = 800)
    
    
    # Undistort
    (w,h) = image.shape[:2]
    newcameramatrix, roi = cv2.getOptimalNewCameraMatrix(CamMat, Distort, (w, h), 0, (w,h))
    dst = cv2.undistort(image, CamMat, Distort, None, newcameramatrix)

    #Crop the frame
    x, y, w, h = roi
    dst = dst[y:y+h, x:x+w]

    corners, ids,_ = cv2.aruco.detectMarkers(dst, aruco_dict, parameters = aruco_param)
    if len(corners) <=0:
        if CACHED_PTS is not None:
            corners = CACHED_PTS
    if len(corners) >0:
        CACHED_PTS = corners
        if ids is not None:
            ids = ids.flatten()
            CACHED_IDS = ids
        else:
            if CACHED_IDS is not None:
                    ids = CACHED_IDS
        if len(corners) < 2:
            if len(CACHED_IDS) >= 2:
                corners = CACHED_PTS
        for (markerCorner, markerID) in zip(corners, ids):
            corner_abcd = markerCorner.reshape((4,2))
            (topL, topR, botR, botL) = corner_abcd
            topLPoint = (int(topL[0]), int(topL[1]))       
            topRPoint = (int(topR[0]), int(topR[1]))
            botLPoint = (int(botL[0]), int(botL[1]))       
            botRPoint = (int(botR[0]), int(botR[1]))

            cv2.line(dst, topLPoint, topRPoint, (0, 0, 255), 2)
            cv2.line(dst, topRPoint, botRPoint, (0, 0, 255), 2)
            cv2.line(dst, botRPoint, botLPoint, (0, 0, 255), 2)
            cv2.line(dst, botLPoint, topLPoint, (0, 0, 255), 2)

            cx = int((topL[0] + botR[0])/2)
            cy = int((topL[1] + botR[1])/2) 

            measure = 3.4 / (topL[0]-cx)
            cv2.circle(dst, (cx, cy), 4, (255, 0, 0),1)
            cv2.putText(dst, str(     
                int(markerID)), (int(topL[0]-10), int(topL[1]-10)), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255))
            Dist.append((cx, cy))
                
            if len(Dist) ==0:
                if Line_Pts is not None:
                    Dist = Line_Pts
            if len(Dist) ==2:
                Line_Pts = Dist
                cv2.line(dst, Dist[0], Dist[1], (255, 0, 0), 2)
                ed = ((Dist[0][0] - Dist[1][0]) ** 2 +
                  ((Dist[0][1] - Dist[1][1]) ** 2)) ** (0.5)
                cv2.putText(dst, str(int(measure * (ed))) + "cm", (int(300), int(300)), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255))
                print("ed", ed*measure)

    cv2.imshow("Marker detected", dst)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
cv2.destroyAllWindows()
vs.stop()
