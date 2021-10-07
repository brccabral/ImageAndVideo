import cv2 as cv
import numpy as np

# img = cv.imread("Resources/Photos/cat.jpg")
# cv.imshow("Cat", img)

# cv.waitKey(0)

# width, height, channels
blank = np.zeros((500,500,3), dtype='uint8')
cv.imshow('Blank', blank)

# paint image with color
#Blue,Green,Red

blank[:] = 0,255,0
cv.imshow('Green', blank)

blank[:] = 0,0,255
cv.imshow('Red', blank)

cv.waitKey(0)