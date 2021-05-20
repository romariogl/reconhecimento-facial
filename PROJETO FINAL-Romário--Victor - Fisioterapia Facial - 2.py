import cv2  # OpenCV
import matplotlib.pyplot as plt  # Matplotlib
import numpy as np  # Numpy
import os, sys
import time

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
var = 0
# Reading vocabulary
fs = cv2.FileStorage('dictionary.yml', cv2.FILE_STORAGE_READ)
voc = fs.getNode('vocabulary').mat()
fs.release()

# BOW
bowDE = cv2.BOWImgDescriptorExtractor(sift, flann)
bowDE.setVocabulary(voc)

# SVM Load
svm = cv2.ml.SVM_load("svm2.xml")

# Fisio Interface
while True:
    print("ogey")
    ret, frame = cap.read()
    if ret == True:
        print("ok")
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
        cv2.imshow('webcam', frame)

    for (x, y, w, h) in faces:
        time.sleep(0.1)
        print('starting image test')
        rdi = gray[y:y + h, x:x + w]
        kp, des = sift.detectAndCompute(rdi, None)
        # img = cv2.drawKeypoints(rdi, kp, frame)
        bdes = bowDE.compute(rdi, kp, des)
        p, s = svm.predict(bdes)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (125, 125, 125), 6)
        font = cv2.FONT_HERSHEY_SIMPLEX
        # Expression Interface
        '''
        if s == 0:
            cv2.putText(frame, 'Anger', (x, 50), font, 1, (125, 0, 125), 5, cv2.LINE_AA)
            cv2.imshow('webcam', frame)
        if s == 1:
            cv2.putText(frame, 'Disgust', (x, 50), font, 1, (125, 0, 125), 5, cv2.LINE_AA)
            cv2.imshow('webcam', frame)
        if s == 2:
            cv2.putText(frame, 'Fear', (x, 50), font, 1, (125, 0, 125), 5, cv2.LINE_AA)
            cv2.imshow('webcam', frame)
        '''
        if s == 3:
            cv2.putText(frame, 'Happy', (x, 50), font, 1, (125, 0, 125), 5, cv2.LINE_AA)
            cv2.imshow('webcam', frame)

        if s == 4:
            cv2.putText(frame, 'Neutral', (x, 50), font, 1, (125, 0, 125), 5, cv2.LINE_AA)
            cv2.imshow('webcam', frame)
        '''
        if s == 5:
            cv2.putText(frame, 'Tristeza', (x, 50), font, 1, (125, 0, 125), 5, cv2.LINE_AA)
            cv2.imshow('webcam', frame)
       
        if s == 6:
            cv2.putText(frame, 'Surpresa', (x, 50), font, 1, (125, 0, 125), 5, cv2.LINE_AA)
            cv2.imshow('webcam', frame)
            var += 1
        '''
        print(p, s, var)
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
