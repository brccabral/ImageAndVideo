import cv2 as cv
import numpy as np

img: np.ndarray = cv.imread('Resources/Photos/cats.jpg')
cv.imshow('Cats', img)

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('Gray', gray)

# Simple thresholding
# THRESH_BINARY = if pixel > 150 set to 255 else 0
threshold, thresh = cv.threshold(gray, 150, 255, cv.THRESH_BINARY)
cv.imshow('Simple thresholding', thresh)

# Inverse thresholding
# THRESH_BINARY_INV = if pixel < 150 set to 255 else 0
threshold, thresh = cv.threshold(gray, 150, 255, cv.THRESH_BINARY_INV)
cv.imshow('Inverse thresholding', thresh)

# Adaptive thresholding
adaptive_thresh = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 11, 1)
cv.imshow('Adaptive thresholding 11 1', adaptive_thresh)

adaptive_thresh = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 13, 1)
cv.imshow('Adaptive thresholding 13 1', adaptive_thresh)

adaptive_thresh = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 11, 3)
cv.imshow('Adaptive thresholding 11 3', adaptive_thresh)

adaptive_thresh = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 1)
cv.imshow('Adaptive thresholding ADAPTIVE_THRESH_GAUSSIAN_C 11 1', adaptive_thresh)

cv.waitKey(0)