import cv2
from cv2 import aruco
import numpy as np
import glob

#Choose dictionary
aruco_dict = cv2.aruco.getPredefinedDictionary( aruco.DICT_7X7_1000)
aruco_para = cv2.aruco.DetectorParameters_create()
#                               Step1
markerX = 4
markerY = 5
length = 3  #cm
sep = .5    #cm

#Print out the marker, 1 is its ID and 400 is image size (pixels)
gridboard = aruco.GridBoard_create(markerX, markerY, length, sep, aruco_dict)

#                               Step2
imgs = glob.glob('./Input_Img/logi/PImg*.jpg')
aruco_corners = []
aruco_ids = []
counter = []
first = True

for fname in imgs:
    img = cv2.imread(fname)
    print("Image size in pixels: ", img.shape[:2])
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # print(fname)
    corner, id, _ = cv2.aruco.detectMarkers(img_gray, aruco_dict, parameters =aruco_para)
    if id is not None:
        cv2.aruco.drawDetectedMarkers(img, corner, id)  
        print('Marker ID: ', id)
        print('Corners coordination: ', corner)
        cv2.imshow("Result", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    if first ==True:
        aruco_corners = corner
        aruco_ids = id
        first = False
    else:
        aruco_corners = np.vstack((aruco_corners, corner))
        aruco_ids = np.vstack((aruco_ids, id))
    counter.append(len(id))
counter = np.array(counter)

    #Append marker data
    # if np.size(corner) == 0:

    # else:
    #     cv2.aruco.drawDetectedMarkers(img, corner, id)
    #     aruco_corners.append(corner)
    #     aruco_ids.append(id)
    #     cv2.imshow("Result", img)
cv2.waitKey(0)
cv2.destroyAllWindows()  
print("Counter:", counter)

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
calibration_flag = cv2.CALIB_RATIONAL_MODEL


# Calibration using ArUco method
print("Object coordinate: ", gridboard)
print("aruco corners: ", aruco_corners)

ret, mtrx, dist, rvect, tvect = cv2.aruco.calibrateCameraAruco(aruco_corners, aruco_ids, counter, gridboard, img_gray.shape, None, None)
print("Camera Matrix is: \n", mtrx)
print("Distortion coefficient is:", dist)

#Save camera matrix and distortion coefficient as npy file
np.save('cameramatrix.npy', mtrx)
np.save('distortion.npy', dist)