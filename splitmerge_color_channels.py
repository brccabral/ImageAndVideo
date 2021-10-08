import cv2 as cv
import numpy as np

img: np.ndarray = cv.imread('Resources/Photos/park.jpg')
cv.imshow('Park', img)

b,g,r = cv.split(img)

cv.imshow('Blue', b)
cv.imshow('Green', g)
cv.imshow('Red', r)

print(img.shape)
print(b.shape)
print(g.shape)
print(r.shape)

cv.waitKey(0)