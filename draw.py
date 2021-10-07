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

# paint a portion of the image
blank[200:300, 300:400] = 255,0,0
cv.imshow('Blue square on Red', blank) # previous blank is Red

# draw rectangle
blank[:]=0,0,0 # reset to black
# np, start, end, color, thickness
cv.rectangle(blank, (0,0), (250,400), (0,255,0), thickness=2)
cv.imshow('Rectangle', blank)

cv.rectangle(blank, (0,0), (250,400), (0,255,0), thickness=cv.FILLED)
cv.imshow('Rectangle filled', blank)

blank[:]=0,0,0 # reset to black
cv.rectangle(blank, (0,0), (blank.shape[1]//2,blank.shape[0]//2), (0,255,0), thickness=cv.FILLED)
cv.imshow('Square', blank)

cv.waitKey(0)