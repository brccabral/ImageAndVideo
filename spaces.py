import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

img: np.ndarray = cv.imread('Resources/Photos/park.jpg')
cv.imshow('Park', img)

# BGR to Gray
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('Gray', gray)

# BGR to HSV
hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
cv.imshow('HSV', hsv)

# BGR to LAB
lab = cv.cvtColor(img, cv.COLOR_BGR2LAB)
cv.imshow('LAB', lab)

# BGR to RGB
rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
cv.imshow('RGB', rgb)

# HSV to BGR
hsv_bgr = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
cv.imshow('HSV to BGR', hsv_bgr)

cv.waitKey(0)

# other libraries use RGB (inverted)
plt.imshow(img)
plt.show()

plt.imshow(rgb)
plt.show()
