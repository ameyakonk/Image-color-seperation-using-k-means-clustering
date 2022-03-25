import numpy as np
import cv2

class CoinDetect:

    def coinSegmentation(self):
        img = cv2.imread('Q1image.png')
        kernel = np.ones((5,5),np.uint8)
        erosion = cv2.erode(img,kernel,iterations = 7) 
        erosion = cv2.dilate(erosion, kernel, iterations=4)
        cv2.imshow("Blobs Using Area", erosion)
        erosion = cv2.bitwise_not(erosion)
       
        params = cv2.SimpleBlobDetector_Params()

        params.filterByArea = True
        params.minArea = 100

        detector = cv2.SimpleBlobDetector_create(params)
        keypoints = detector.detect(erosion)
        print("No. of coins detected: ",len(keypoints))
        erosion = cv2.bitwise_not(erosion)
        blank = np.zeros((1, 1))
        blobs = cv2.drawKeypoints(erosion, keypoints, blank, (0, 0, 255),cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        cv2.imshow("Blobs Using Area", blobs)
        #cv2.imshow("frame", im_with_keypoints)
        cv2.waitKey(0) # waits until a key is pressed
        cv2.destroyAllWindows()        

p = CoinDetect()
p.coinSegmentation()