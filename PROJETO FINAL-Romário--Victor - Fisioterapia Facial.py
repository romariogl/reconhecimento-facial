import cv2  # OpenCV
import matplotlib.pyplot as plt  # Matplotlib
import numpy as np  # Numpy
import os, sys

# Video and face
cap = cv2.VideoCapture('teste fisio.mp4')  # Video variable
face_cascade = cv2.CascadeClassifier('class/haarcascade_frontalface_default.xml')  # Face detecting variable

# SIFT variable
sift = cv2.xfeatures2d.SIFT_create()

# FLANN PARAMETERS
FLANN_INDEX_KDTREE = 1  # FLANN_INDEX_KMEANS = 2
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=10)  # trees = 5
search_params = dict(checks=50)  # checks = 50, or pass empty dictionary
flann = cv2.FlannBasedMatcher(index_params, search_params)

# Reading vocabulary
fs = cv2.FileStorage('dictionary.yml', cv2.FILE_STORAGE_READ)
voc = fs.getNode('vocabulary').mat()
fs.release()

# BOW
bowDE = cv2.BOWImgDescriptorExtractor(sift, flann)
bowDE.setVocabulary(voc)

# SVM Load
svm = cv2.ml.SVM_load("svm.xml")
var = 0
# Fisio Interface
while True:
    #print("ogey")
    ret, frame = cap.read()
    if ret == True:
        #print("ok")
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
        #cv2.imshow('webcam', frame)
        #print('starting image test')
        kp, des = sift.detectAndCompute(gray, None)
        # img = cv2.drawKeypoints(rdi, kp, frame)
        bdes = bowDE.compute(gray, kp, des)
        p, s = svm.predict(bdes)
        # Expression Interface

        if s == 3:
            print('happy')

        if s == 6:
           print('surprise')

##        print(p, s)


    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

# while True:
#    print("ogey")
#    ret, frame = cap.read()
#    copia = frame.copy()
#    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)


#    for (x, y, w, h) in faces:
#        print('aq')
#        rdi = gray[y:y + 256, x:x + 256]
#        rdi_color = copia[y:y + h, x:x + w]
#        kp, des = sift.detectAndCompute(rdi, None)
#        img = cv2.drawKeypoints(rdi, kp, frame)
#        bdes = bowDE.compute(rdi, kp)
#        cv2.imshow('gaypoints', img)
#        print(rdi.shape)

#    key = cv2.waitKey(1) & 0xFF
#    if key == ord("q"):
#        break

# cap.release()
# cv2.destroyAllWindows()
