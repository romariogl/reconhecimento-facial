import cv2  # OpenCV
import matplotlib.pyplot as plt  # Matplotlib
import numpy as np  # Numpy
import os, sys

cap = cv2.VideoCapture(0)  # Video variable
face_cascade = cv2.CascadeClassifier('class/haarcascade_frontalface_default.xml')  # Face detecting variable
sift = cv2.xfeatures2d.SIFT_create()  # SIFT variable

# Reading vocabulary
fs = cv2.FileStorage('dictionary.yml', cv2.FILE_STORAGE_READ)
voc = fs.getNode('vocabulary').mat()
fs.release()


#fs = open("dictionary.yml", "r")  # Open BOW Dictionary file
#print(fs.read())    # Read dictionary file


# FLANN PARAMETERS
FLANN_INDEX_KDTREE = 1 # FLANN_INDEX_KMEANS = 2
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 10) # trees = 5
search_params = dict(checks=50)   #checks = 50, or pass empty dictionary
flann = cv2.FlannBasedMatcher(index_params,search_params)

# BOW
bowDE = cv2.BOWImgDescriptorExtractor(sift, flann)
bowDE.setVocabulary(voc)
print(voc.shape)
# Variable to train data
traindata = []
trainlabels = []
bowsig = []
var = 0
var2 = 0

# Diretórios das imagens
dir0 = os.listdir('/DATA/treino 2400/anger')
dir1 = os.listdir('/DATA/treino 2400/disgust')
dir2 = os.listdir('/DATA/treino 2400/fear')
dir3 = os.listdir('/DATA/treino 2400/happy')
dir4 = os.listdir('/DATA/treino 2400/neutral')
dir5 = os.listdir('/DATA/treino 2400/sadness')
dir6 = os.listdir('/DATA/treino 2400/surprise')

# Reading images
'''
print('0')
for filename in dir0:
    input = cv2.imread('/DATA/treino 2400/anger/{}' .format(filename), 0)
    #print('0')
    #print(filename)
    kp, des = sift.detectAndCompute(input, None)
    bowsig = bowDE.compute(input, kp, des)
    #print(bowsig)
    traindata.extend(bowsig)
    trainlabels.append(0)

print('1')
for filename in dir1:
    #print('1')
    #print(filename)
    input = cv2.imread('/DATA/treino 2400/disgust/{}' .format(filename), 0)
    kp, des = sift.detectAndCompute(input, None)
    bowsig = bowDE.compute(input, kp, des)
    traindata.extend(bowsig)
    trainlabels.append(1)

print('2')
for filename in dir2:
    #print('2')
    #print(filename)
    input = cv2.imread('/DATA/treino 2400/fear/{}' .format(filename), 0)
    kp, des = sift.detectAndCompute(input, None)
    bowsig = bowDE.compute(input, kp, des)
    traindata.extend(bowsig)
    trainlabels.append(2)

'''
print('3')
for filename in dir3:
    #print('3')
    #print(filename)
    input = cv2.imread('/DATA/treino 2400/happy/{}' .format(filename), 0)
    kp, des = sift.detectAndCompute(input, None)
    bowsig = bowDE.compute(input, kp, des)
    traindata.extend(bowsig)
    trainlabels.append(3)


print('4')
for filename in dir4:
    #print('4')
    #print(filename)
    input = cv2.imread('/DATA/treino 2400/neutral/{}' .format(filename), 0)
    kp, des = sift.detectAndCompute(input, None)
    bowsig = bowDE.compute(input, kp, des)
    traindata.extend(bowsig)
    trainlabels.append(4)
'''
print('5')
for filename in dir5:
    #print('5')
    #print(filename)
    input = cv2.imread('/DATA/treino 2400/sadness/{}' .format(filename), 0)
    kp, des = sift.detectAndCompute(input, None)
    bowsig = bowDE.compute(input, kp, des)
    traindata.extend(bowsig)
    trainlabels.append(5)

print('6')
for filename in dir6:
    #print('6')
    #print(filename)
    input = cv2.imread('/DATA/treino 2400/surprise/{}' .format(filename), 0)
    kp, des = sift.detectAndCompute(input, None)
    bowsig = bowDE.compute(input, kp, des)
    traindata.extend(bowsig)
    trainlabels.append(6)

'''


# Creating and training SVM
print('creating svm')
svm = cv2.ml.SVM_create()
svm.setKernel(cv2.ml.SVM_RBF)
svm.setType(cv2.ml.SVM_NU_SVC)
svm.setNu(0.5) ## old = 0.5
svm.setTermCriteria((3, 1000, 0.01)) #mexido
svm.setGamma(100) # old = 10


svm.setC(100) # Optimization value altered when using C_SVC type
svm.setDegree(1) # Value altered when using poly kernel
#svm.setP(0) ## Value altered when using EPS_SVR type
#svm.setCoef0(0.01) # Value altered when using sigmoid kernel


print('training svm')
print(np.array(traindata).shape)
print(np.array(trainlabels).shape)
svm.train(np.array(traindata), cv2.ml.ROW_SAMPLE, np.array(trainlabels))
svm.save("svm2.xml")


#svm = cv2.ml.SVM_create()
#svm.train(np.array(traindata), cv2.ml.ROW_SAMPLE, np.array(trainlabels))



#Starting training error data gathering
'''
for filename in dir0:
    input = cv2.imread('/DATA/treino 2400/anger/{}' .format(filename), 0)
    kp, des = sift.detectAndCompute(input, None)
    bowsig2 = bowDE.compute(input, kp, des)
    p, s = svm.predict(bowsig2)
    if int(s) == 0:
        var += 1


print(var)
for filename in dir1:
    input = cv2.imread('/DATA/treino 2400/disgust/{}' .format(filename), 0)
    kp, des = sift.detectAndCompute(input, None)
    bowsig2 = bowDE.compute(input, kp, des)
    p, s = svm.predict(bowsig2)
    if int(s) == 1:
        var += 1


print(var)
for filename in dir2:
    input = cv2.imread('/DATA/treino 2400/fear/{}' .format(filename), 0)
    kp, des = sift.detectAndCompute(input, None)
    bowsig2 = bowDE.compute(input, kp, des)
    p, s = svm.predict(bowsig2)
    if int(s) == 2:
        var += 1

print(var)
'''
for filename in dir3:
    input = cv2.imread('/DATA/treino 2400/happy/{}' .format(filename), 0)
    kp, des = sift.detectAndCompute(input, None)
    bowsig2 = bowDE.compute(input, kp, des)
    p, s = svm.predict(bowsig2)
    if int(s) == 3:
        var += 1

print(var)

for filename in dir4:
    input = cv2.imread('/DATA/treino 2400/neutral/{}' .format(filename), 0)
    kp, des = sift.detectAndCompute(input, None)
    bowsig2 = bowDE.compute(input, kp, des)
    p, s = svm.predict(bowsig2)
    if int(s) == 4:
        var += 1

print(var)
'''
for filename in dir5:
    input = cv2.imread('/DATA/treino 2400/sadness/{}' .format(filename), 0)
    kp, des = sift.detectAndCompute(input, None)
    bowsig2 = bowDE.compute(input, kp, des)
    p, s = svm.predict(bowsig2)
    if int(s) == 5:
        var += 1

print(var)

for filename in dir6:
    input = cv2.imread('/DATA/treino 2400/surprise/{}' .format(filename), 0)
    kp, des = sift.detectAndCompute(input, None)
    bowsig2 = bowDE.compute(input, kp, des)
    p, s = svm.predict(bowsig2)
    if int(s) == 6:
        var += 1
print(var)
'''

#Starting test data gathering
dir0 = os.listdir('/DATA/teste 2400/angry')
dir1 = os.listdir('l/DATA/teste 2400/disgust')
dir2 = os.listdir('DATA/teste 2400/fear')
dir3 = os.listdir('/DATA/teste 2400/happy')
dir4 = os.listdir('/DATA/teste 2400/neutral')
dir5 = os.listdir('/DATA/teste 2400/sad')
dir6 = os.listdir('/DATA/teste 2400/surprise')
'''
for filename in dir0:
    input = cv2.imread('/DATA/teste 2400/angry/{}' .format(filename), 0)
    kp, des = sift.detectAndCompute(input, None)
    bowsig2 = bowDE.compute(input, kp, des)
    p, s = svm.predict(bowsig2)
    #print(int(s))
    if int(s) == 0:
        var2 += 1

print(var2)
for filename in dir1:
    input = cv2.imread('/DATA/teste 2400/disgust/{}' .format(filename), 0)
    kp, des = sift.detectAndCompute(input, None)
    bowsig2 = bowDE.compute(input, kp, des)
    p, s = svm.predict(bowsig2)
    if int(s) == 1:
        var2 += 1

print(var2)
for filename in dir2:
    input = cv2.imread('/DATA/teste 2400/fear/{}' .format(filename), 0)
    kp, des = sift.detectAndCompute(input, None)
    bowsig2 = bowDE.compute(input, kp, des)
    p, s = svm.predict(bowsig2)
    if int(s) == 2:
        var2 += 1

print(var2)
'''
for filename in dir3:
    input = cv2.imread('/DATA/teste 2400/happy/{}'.format(filename), 0)
    kp, des = sift.detectAndCompute(input, None)
    bowsig2 = bowDE.compute(input, kp, des)
    p, s = svm.predict(bowsig2)
    if int(s) == 3:
        var2 += 1

print(var2)

for filename in dir4:
    input = cv2.imread('/DATA/teste 2400/neutral/{}' .format(filename), 0)
    kp, des = sift.detectAndCompute(input, None)
    bowsig2 = bowDE.compute(input, kp, des)
    p, s = svm.predict(bowsig2)
    if int(s) == 4:
        var2 += 1

print(var2)
'''
for filename in dir5:
    input = cv2.imread('/DATA/teste 2400/sad/{}'.format(filename), 0)
    kp, des = sift.detectAndCompute(input, None)
    bowsig2 = bowDE.compute(input, kp, des)
    p, s = svm.predict(bowsig2)
    if int(s) == 5:
        var2 += 1

print(var2)

for filename in dir6:
    input = cv2.imread('/DATA/teste 2400/surprise/{}'.format(filename), 0)
    kp, des = sift.detectAndCompute(input, None)
    bowsig2 = bowDE.compute(input, kp, des)
    p, s = svm.predict(bowsig2)
    if int(s) == 6:
        var2 += 1

print(var2)
'''
'''print(100*var/10369)
print(100*var2/1307)'''

