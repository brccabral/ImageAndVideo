import cv2 as cv
import numpy as np

img: np.ndarray = cv.imread('Resources/Photos/cats.jpg')
cv.imshow('Cats', img)

blank = np.zeros(img.shape[:2], dtype='uint8')
circle = cv.circle(blank.copy(), (img.shape[1]//2+60,img.shape[0]//2), 100, 255, -1)
cv.imshow('Mask circle', circle)

masked = cv.bitwise_and(img, img, mask=circle)
cv.imshow('Masked Circle Image', masked)

rectangle = cv.rectangle(blank.copy(), (img.shape[1]//2-60,img.shape[0]//2), (img.shape[1]//2+70,img.shape[0]//2+100), 255, -1)
cv.imshow('Mask rectangle', rectangle)

masked = cv.bitwise_and(img, img, mask=rectangle)
cv.imshow('Masked Rectangle Image', masked)

weird_shape = cv.bitwise_and(rectangle, circle)
cv.imshow('Mask weird', circle)
masked = cv.bitwise_and(img, img, mask=weird_shape)
cv.imshow('Masked Weird Image', masked)

cv.waitKey(0)