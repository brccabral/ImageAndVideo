import cv2 as cv
import numpy as np

img: np.ndarray = cv.imread('Resources/Photos/lady.jpg')
cv.imshow('Person', img)

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('Gray', gray)

cv.waitKey(0)