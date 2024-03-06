import cv2
from cv2 import aruco
import numpy as np

cap = cv2.VideoCapture(0)

#Aruco information:
aruco_dict = cv2.aruco.getPredefinedDictionary(aruco.DICT_7X7_100)
aruco_para = cv2.aruco.DetectorParameters_create()
CamMat = np.load('cameramatrix.npy')
Distort = np.load('distortion.npy')
length = 4


while True:

    ret, frame = cap.read()
    (w,h) = frame.shape[:2]
    NewCamMat, roi = cv2.getOptimalNewCameraMatrix(CamMat, Distort, (w, h), 0, (w,h))
    frame = cv2.undistort(frame, CamMat, Distort, None, NewCamMat)

    corner, id, _= cv2.aruco.detectMarkers(frame, aruco_dict, parameters = aruco_para)
    if len(corner) >0:
        for i in range(0, len(id)):
            rvec, tvec, _o= cv2.aruco.estimatePoseSingleMarkers(corner[i], length, CamMat, Distort)
            cv2.aruco.drawDetectedMarkers(frame, corner, id)
            cv2.drawFrameAxes(frame, CamMat, Distort, rvec, tvec, 3)
            print("Rotation vector: ", rvec)
            # print("Translation vector: ", tvec)
    #Display result
    cv2.imshow("Result", frame)
    if cv2.waitKey(1) & 0XFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
