import cv2 as cv
import numpy as np

img: np.ndarray = cv.imread("Resources/Photos/park.jpg")
cv.imshow("Park", img)

b, g, r = cv.split(img)

cv.imshow("Blue", b)
cv.imshow("Green", g)
cv.imshow("Red", r)

print(img.shape)
print(b.shape)
print(g.shape)
print(r.shape)

merged = cv.merge([b, g, r])
cv.imshow("Merged", merged)

blank = np.zeros(img.shape[:2], dtype="uint8")
blue = cv.merge([b, blank, blank])
green = cv.merge([blank, g, blank])
red = cv.merge([blank, blank, r])

cv.imshow("Blue channel", blue)
cv.imshow("Green channel", green)
cv.imshow("Red channel", red)

cv.waitKey(0)
