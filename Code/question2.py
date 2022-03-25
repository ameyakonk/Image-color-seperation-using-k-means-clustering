import cv2
import numpy as np
import sys

class Image_Stitching():
    def featuredetection(self):
        MIN_MATCH_COUNT = 10
        img1Color = cv2.imread('Q2imageA.png')  
        img2Color = cv2.imread('Q2imageB.png')
        # convert images to grayscale
        img1 = cv2.cvtColor(img1Color, cv2.COLOR_BGR2GRAY)
        img2 = cv2.cvtColor(img2Color, cv2.COLOR_BGR2GRAY)
        # create SIFT object
        sift = cv2.xfeatures2d.SIFT_create()
        # detect SIFT features in both images
        keypoints_1, descriptors_1 = sift.detectAndCompute(img1,None)
        keypoints_2, descriptors_2 = sift.detectAndCompute(img2,None)
        # create feature matcher
        bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)
        # match descriptors of both images
        matches = bf.match(descriptors_1,descriptors_2)
        matches = sorted(matches, key = lambda x:x.distance)
  
        src_pts = np.float32([ keypoints_1[m.queryIdx].pt for m in matches ]).reshape(-1,1,2)
        dst_pts = np.float32([ keypoints_2[m.trainIdx].pt for m in matches ]).reshape(-1,1,2)

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
        #matchesMask = mask.ravel().tolist()
        width = img1.shape[0] + img2.shape[0]
        height = img1.shape[1] + img2.shape[1]
        im_out = cv2.warpPerspective(img1Color, M, (width, height))
        im_out[0:img2Color.shape[0], 0:img2Color.shape[1]] = img2Color
        im_out = im_out[0:300, 0:400]
        # draw first 50 matches
        cv2.imshow('image', im_out)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        # save the image
        
        
p = Image_Stitching()
p.featuredetection()