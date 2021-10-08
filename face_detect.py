import cv2 as cv
import numpy as np

img: np.ndarray = cv.imread('Resources/Photos/lady.jpg')
cv.imshow('Person', img)

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('Gray', gray)

haar_cascade: cv.CascadeClassifier = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3)
print(f'Number of faces found = {len(faces_rect)}')

for (x,y,w,h) in faces_rect:
    cv.rectangle(img, (x,y), (x+w,y+h), (0,255,0), thickness=2)
cv.imshow('Detected faces', img)

img: np.ndarray = cv.imread('Resources/Photos/group 2.jpg')
cv.imshow('Group', img)

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('Group Gray', gray)

haar_cascade: cv.CascadeClassifier = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3)
print(f'Number of faces found = {len(faces_rect)}')

for (x,y,w,h) in faces_rect:
    cv.rectangle(img, (x,y), (x+w,y+h), (0,255,0), thickness=2)
cv.imshow('Group Detected faces', img)

cv.waitKey(0)