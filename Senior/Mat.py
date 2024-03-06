import cv2
import math
import numpy as np

# Function to calculate roll, pitch, yaw in degree
def rvec2rpy(rvec):
    rmat, _ = cv2.Rodrigues(rvec)
    roll = math.atan2(rmat[2][1], rmat[2][2])
    roll_deg = roll*180/math.pi
    yaw  = math.atan2(rmat[1][0], rmat[0][0])
    yaw_deg = yaw*180/math.pi
    if math.cos(yaw) == 0:
        pitch = math.atan2(-rmat[2][0], rmat[1][0]/math.sin(yaw))
        pitch_deg = pitch*180/math.pi
    else:
        pitch = math.atan2(-rmat[2][0], rmat[0][0]/math.cos(yaw))
        pitch_deg = pitch*180/math.pi
    return(roll_deg, pitch_deg, yaw_deg)

# Function to determine the homogeneous transformation matrix
def homoTransMat(rvec, tvec):
    rmat, _ = cv2.Rodrigues(rvec)
    TransMat = np.array([[rmat[0,0], rmat[0, 1], rmat[0,2], tvec[0,0]],
            [rmat[1,0], rmat[1, 1], rmat[1,2], tvec[0,1]],
            [rmat[2,0], rmat[2, 1], rmat[2,2], tvec[0,2]],
            [0         ,    0       ,   0       ,   1       ] ])
    return(TransMat)

# Function to calculate distance of 2 points in 3D:
def distance(tvec1, tvec2):
    dist = math.sqrt((tvec1[0,0] - tvec2[0,0])**2 + (tvec1[0,1] - tvec2[0,1])**2 + (tvec1[0,2] - tvec2[0,2])**2)
    return(dist)

# Function to determine relative position of marker B wrt marker A:
def relpos(Cam_T_A, Cam_T_B):

    A_T_Cam = np.linalg.inv(Cam_T_A)
    A_T_B = np.dot(A_T_Cam, Cam_T_B)
    rmat_ATB = np.array([[A_T_B[0,0], A_T_B[0, 1], A_T_B[0,2]],
            [A_T_B[1,0], A_T_B[1, 1], A_T_B[1,2]],
            [A_T_B[2,0], A_T_B[2, 1], A_T_B[2,2]] ])
    
    rvec_ATB,_ = cv2.Rodrigues(rmat_ATB)
    tvec_ATB = np.array([[A_T_B[0,3], A_T_B[1,3], A_T_B[2,3]]])
    return(A_T_B,rvec_ATB, tvec_ATB)

# def invRT(rvec, tvec):
#     rmat, _ = cv2.Rodrigues(rvec)
#     rmat_transposed = np.matrix(rmat).T
#     invTvec = np.dot(-rmat_transposed, np.matrix(tvec))
#     invRvec,_ = cv2.Rodrigues(rmat_transposed)
#     return(invRvec, invTvec)
    
# # relative position of marker 2 wrt marker 1
# def relativePose(rvec1, tvec1, rvec2, tvec2):
#     rvec1, tvec1 = rvec1.reshape((3,1)), tvec1.reshape((3,1))
#     rvec2, tvec2 = rvec2.reshape((3,1)), tvec2.reshape((3,1))

#     invRvec1, invTvec1 = invRT(rvec1, tvec1)
#     info = cv2.composeRT(rvec2, tvec2, invRvec1, invTvec1)
#     composedRvec, composedTvec = info[0], info[1]
#     composedRvec = composedRvec.reshape((3, 1))
#     composedTvec = composedTvec.reshape((3, 1))
#     return composedRvec, composedTvec

    