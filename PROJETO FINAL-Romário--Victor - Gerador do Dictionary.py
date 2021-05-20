import cv2 # OpenCV
import matplotlib.pyplot as plt # Matplotlib
import numpy as np # Numpy
import glob
import os, sys

face_cascade = cv2.CascadeClassifier('class/haarcascade_frontalface_default.xml')
sift = cv2.xfeatures2d.SIFT_create()
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
featuresUnclustered=None
dir1 = os.listdir('/DATA/tudo 2400')

input = np.zeros((48, 48))
bowTrainer = cv2.BOWKMeansTrainer(10000, (3, 10000,0.00001), 1, cv2.KMEANS_PP_CENTERS)
bowTrainer.clear()
var = 0


for filename in dir1:
    var += 1
    print('{}' .format(var))
    input = cv2.imread('/tudo 2400/{}' .format(filename), 0)
    kp, des = sift.detectAndCompute(input, None)
    if des is not None:
        #print('entrei aq')
        if featuresUnclustered is None:
            featuresUnclustered = des
        else:
            featuresUnclustered = np.append(featuresUnclustered, des, axis=0)

print('starting BOW add')
bowTrainer.add(featuresUnclustered)
print('starting BOW cluster')
Dictionary = bowTrainer.cluster()
fs = cv2.FileStorage('dictionary.yml', flags = 1)
fs.write(name = 'vocabulary', val = Dictionary )
fs.release()