import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

img: np.ndarray = cv.imread('Resources/Photos/cats.jpg')
cv.imshow('Cats', img)

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('Gray', gray)

# Grayscale histogram
gray_hist = cv.calcHist([gray], [0], mask=None, histSize=[256], ranges=[0,256])

plt.figure()
plt.title('Grayscale Histogram')
plt.xlabel('Bins')
plt.ylabel('# of pixels')
plt.plot(gray_hist)
plt.xlim([0,256])
plt.show()

blank = np.zeros(img.shape[:2], dtype='uint8')
circle = cv.circle(blank.copy(), (img.shape[1]//2,img.shape[0]//2), 100, 255, -1)
cv.imshow('Mask circle', circle)

masked = cv.bitwise_and(gray, gray, mask=circle)
cv.imshow('Masked Circle Image', masked)

# Grayscale masked histogram
gray_hist = cv.calcHist([gray], [0], mask=circle, histSize=[256], ranges=[0,256])

plt.figure()
plt.title('Grayscale circle Histogram')
plt.xlabel('Bins')
plt.ylabel('# of pixels')
plt.plot(gray_hist)
plt.xlim([0,256])
plt.show()

# Color histogram
plt.figure()
colors = ('b', 'g', 'r')
for i, color in enumerate(colors):
    hist = cv.calcHist([img], [i], None, [256], [0,256])
    plt.plot(hist, color=color)
    plt.xlim([0,256])
plt.title('Color Histogram')
plt.xlabel('Bins')
plt.ylabel('# of pixels')
plt.show()
plt.show()

# Color mask histogram
plt.figure()
colors = ('b', 'g', 'r')
for i, color in enumerate(colors):
    hist = cv.calcHist([img], [i], circle, [256], [0,256])
    plt.plot(hist, color=color)
    plt.xlim([0,256])
plt.title('Color Histogram')
plt.xlabel('Bins')
plt.ylabel('# of pixels')
plt.show()

cv.waitKey(0)