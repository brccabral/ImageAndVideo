import cv2 as cv
import numpy as np

img: np.ndarray = cv.imread('Resources/Photos/cats.jpg')
cv.imshow('Cats', img)

blank = np.zeros(img.shape[:2], dtype='uint8')
mask = cv.circle(blank.copy(), (img.shape[1]//2+60,img.shape[0]//2), 100, 255, -1)
cv.imshow('Mask circle', mask)

masked = cv.bitwise_and(img, img, mask=mask)
cv.imshow('Masked Circle Image', masked)

mask = cv.rectangle(blank.copy(), (img.shape[1]//2-60,img.shape[0]//2), (img.shape[1]//2+70,img.shape[0]//2+100), 255, -1)
cv.imshow('Mask rectangle', mask)

masked = cv.bitwise_and(img, img, mask=mask)
cv.imshow('Masked Rectangle Image', masked)

cv.waitKey(0)