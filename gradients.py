import cv2 as cv
import numpy as np

img: np.ndarray = cv.imread('Resources/Photos/cats.jpg')
cv.imshow('Cats', img)

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('Gray', gray)

# Laplacian
lap = cv.Laplacian(gray, cv.CV_64F) # this computes differences
lap = np.uint8(np.absolute(lap)) # images can't have negative numbers
cv.imshow('Laplacian', lap)

cv.waitKey(0)