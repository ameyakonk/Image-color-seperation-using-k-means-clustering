from hashlib import new
import numpy as np
import cv2
import math 
from operator import itemgetter
import matplotlib.pyplot as plt

class CoinDetect:

    def eucDistance(self, img1, img2):
        return math.pow(math.pow((img1[0]-img2[0]),2) + math.pow((img1[1]-img2[1]),2) + math.pow((img1[2]-img2[2]),2), 0.5)


    def calculateMeanError(self, list1_, list2_):
        list1 = list1_.copy()
        list2 = list2_.copy()

        threshold = 0.1
        a1,b1,c1 = map(itemgetter(0),list1), map(itemgetter(1),list1), map(itemgetter(2),list1)
        a1,b1,c1 = map(list,zip(*list1))
        a1 = np.asarray(a1)
        b1 = np.asarray(b1)
        c1 = np.asarray(c1)
        
        a2,b2,c2 = map(itemgetter(0),list2), map(itemgetter(1),list2), map(itemgetter(2),list2)
        a2,b2,c2 = map(list,zip(*list2))
        a2 = np.asarray(a2)
        b2 = np.asarray(b2)
        c2 = np.asarray(c2)
       
        return  sum(np.abs(np.subtract(a1,a2))) < threshold and sum(np.abs(np.subtract(b1,b2))) < threshold and sum(np.abs(np.subtract(c1,c2))) < threshold

    def kMeans(self):
        img = cv2.imread('Q4image.png') 
        r, c,_ = img.shape
        k = 4
        meanList = [[0,0,0], [63,63,63],[127,127,127], [255,255,255]]
        prevMean = meanList
        classList = [[], [], [], []]
        classDict = {}
        newMean = [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]]
        
        while not self.calculateMeanError(prevMean, newMean): 
            classDict.clear()
            prevMean = newMean.copy()
            
            for i in range(r):
                for j in range(c):
                    classFinder = []
                    for data in meanList:
                        classFinder.append(self.eucDistance(img[i, j], data))

                    index = classFinder.index(min(classFinder))
                    classFinder.clear()
                    classList[index].append(img[i, j])
                    classDict[(j, i)] = index
            
            for i in range(k):  
                if len(classList[i]) > 0:
                    a,b,c_ = map(itemgetter(0),classList[i]), map(itemgetter(1),classList[i]), map(itemgetter(2),classList[i])
                    a,b,c_ = map(list,zip(*classList[i]))
                    newMean[i] = [sum(a)/len(a), sum(b)/len(b), sum(c_)/len(c_)]
                    classList[i].clear()
            
            print(newMean)
            meanList = newMean.copy()
        
   
        newImg = np.zeros((r,c,3), np.uint8)
        for i in range(r):
                for j in range(c):
                    newImg[i, j] = newMean[classDict[(j, i)]]
      
        cv2.imshow("frame", newImg)
        cv2.imshow("frame2", img)
        
        cv2.waitKey(0) # waits until a key is pressed
        cv2.destroyAllWindows() 

p = CoinDetect()
p.kMeans()