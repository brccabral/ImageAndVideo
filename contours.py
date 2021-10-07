import cv2 as cv
import numpy as np


img = cv.imread('Resources/Photos/cats.jpg')
cv.imshow('Cats', img)

# convert to grayscale
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('Gray', gray)

# Edge Cascade
canny = cv.Canny(img, 125, 175)
cv.imshow('Canny Edges', canny)

# cv.RETR_EXTERNAl = returns only the outside contours
# cv.RETR_TREE = returns all that are in a hierachal
# cv.RETR_LIST = all
# cv.CHAIN_APPROX_NONE = all
# cv.CHAIN_APPROX_SIMPLE = a line is only the two points
# hierarchies = if one shape is inside another
contours, hierarchies = cv.findContours(canny, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
print(f'{len(contours)} contours(s) found CHAIN_APPROX_SIMPLE')
contours, hierarchies = cv.findContours(canny, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
print(f'{len(contours)} contours(s) found CHAIN_APPROX_NONE')

# blur
blur = cv.GaussianBlur(gray, (5,5), cv.BORDER_DEFAULT)
cv.imshow('Blur', blur)
canny = cv.Canny(blur, 125, 175)
cv.imshow('Canny Edges Blur', canny)
contours, hierarchies = cv.findContours(canny, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
print(f'{len(contours)} contours(s) found blur CHAIN_APPROX_SIMPLE')


# cv.THRESH_BINARY = below 125 it is set to 0, above it is set to 255
ret, thresh = cv.threshold(gray, 125, 255, cv.THRESH_BINARY)
cv.imshow('Threshold', thresh)

contours, hierarchies = cv.findContours(thresh, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
print(f'{len(contours)} contours(s) found thresh CHAIN_APPROX_SIMPLE')


cv.waitKey(0)