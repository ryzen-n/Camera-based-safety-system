import cv2
import numpy as np
import math

rot_mat = np.array([[0.866, -0.5, 0],  
            [0.5,   0.866,  0],
            [0,     0,      1]])
rot_vector, _ = cv2.Rodrigues(rot_mat)
print("Rotation vector: ", rot_mat)