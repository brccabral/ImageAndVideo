import cv2 as cv
import numpy as np

img: np.ndarray = cv.imread('Resources/Photos/park.jpg')
cv.imshow('Park', img)

# BGR to Gray
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('Gray', gray)

cv.waitKey(0)