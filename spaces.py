import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

img: np.ndarray = cv.imread("Resources/Photos/park.jpg")
cv.imshow("Park", img)

# BGR to Gray
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow("Gray", gray)

# BGR to HSV
hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
cv.imshow("HSV", hsv)

# BGR to LAB
lab = cv.cvtColor(img, cv.COLOR_BGR2LAB)
cv.imshow("LAB", lab)

# BGR to RGB
rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
cv.imshow("RGB", rgb)

# HSV to BGR
hsv_bgr = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
cv.imshow("HSV to BGR", hsv_bgr)

# LAB to BGR
lab_bgr = cv.cvtColor(lab, cv.COLOR_LAB2BGR)
cv.imshow("LAB to BGR", lab_bgr)

# Gray to BGR to LAB doesn't look good
gray_bgr = cv.cvtColor(gray, cv.COLOR_GRAY2BGR)
gray_bgr_lab = cv.cvtColor(gray_bgr, cv.COLOR_BGR2LAB)
cv.imshow("Gray to BGR to LAB", gray_bgr_lab)

cv.waitKey(0)

# other libraries use RGB (inverted)
plt.imshow(img)
plt.show()

plt.imshow(rgb)
plt.show()
