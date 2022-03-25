import cv2
import numpy as np
import sys

class CameraCaliberation:
    def projectMatrix(self):
        U = [[757, 213, 0, 0, 0],
                [758, 415,0, 3, 0],
                [758, 686,0, 7, 0],
                [759, 966,0, 11, 0],
                [1190, 172,7, 1, 0],
                [329, 1041,0, 11, 7],
                [1204, 850,7, 9, 0],
                [340, 159,0, 1, 7]]
        A = []
     
        for i in range(8):
            A.append([[U[i][2], U[i][3], U[i][4], 1, 0, 0, 0, 0, (-U[i][0]*U[i][2]), (-U[i][0]*U[i][3]), (-U[i][0]*U[i][4]), -U[i][0]],
                      [0, 0, 0, 0, U[i][2], U[i][3], U[i][4], 1, (-U[i][1]*U[i][2]), (-U[i][1]*U[i][3]), (-U[i][1]*U[i][4]), -U[i][1]]])
        
        A = np.array(A)
        A = np.reshape(A, (16, 12))
        print(A)
        u, s, vh = np.linalg.svd(A)
        P = vh[-1]
        P = np.divide(P, P[-1])
        P = np.reshape(P, (3,4))
        print(P)
        M = P[:,0:3]
        print(M)

        Q, R = np.linalg.qr(np.rot90(M,3))
        R = np.transpose(np.rot90(R,2))
        R = np.divide(R, R[-1][-1])
        print("K Matrix:")
        print(R)

p = CameraCaliberation()
p.projectMatrix()