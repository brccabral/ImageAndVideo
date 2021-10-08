import cv2 as cv
import numpy as np

img: np.ndarray = cv.imread('Resources/Photos/cats.jpg')
cv.imshow('Cats', img)

# Averaging
# in a 3x3 square of pixels
# center pixel is the average of all 8 neighbors
average = cv.blur(img, (3,3))
cv.imshow('Average Blur 3x3', average)

# 7x7 square
average = cv.blur(img, (7,7))
cv.imshow('Average Blur 7x7', average)



cv.waitKey(0)