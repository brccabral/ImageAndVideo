import cv2 as cv
import numpy as np


haar_cascade: cv.CascadeClassifier = cv.CascadeClassifier('haarcascade_frontalface_default.xml')

features = np.load('features.npy', allow_pickle=True)
labels = np.load('labels.npy')

face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.read('face_trained.yml')

people = ['Ben Afflek', 'Elton John', 'Jerry Seinfield', 'Madonna', 'Mindy Kaling']

img = cv.imread(r'Resources/Faces/val/ben_afflek/2.jpg')
cv.imshow('Ben', img)

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('Ben Gray', gray)


cv.waitKey(0)