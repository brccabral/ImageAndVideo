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

# Gaussian Blur
# 7x7, standard deviation 0
# product deviation method
# less blur than avg
gauss = cv.GaussianBlur(img, (7,7), 0)
cv.imshow('Gaussian Blur 7x7 std 0', gauss)
gauss = cv.GaussianBlur(img, (3,3), 0)
cv.imshow('Gaussian Blur 3x3 std 0', gauss)

# Median Blur
median = cv.medianBlur(img, 7)
cv.imshow('Median Blur 7x7', median)
median = cv.medianBlur(img, 3)
cv.imshow('Median Blur 3x3', median)

# Bilateral
# keeps edges
# np, diameter, how far away a pixel can influence in calculations
bilateral = cv.bilateralFilter(img, 10, 35, 25)
cv.imshow('Bilateral', bilateral)

cv.waitKey(0)