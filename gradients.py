import cv2 as cv
import numpy as np

img: np.ndarray = cv.imread("Resources/Photos/cats.jpg")
cv.imshow("Cats", img)

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow("Gray", gray)

# Laplacian
lap = cv.Laplacian(gray, cv.CV_64F)  # this computes differences
lap = np.uint8(np.absolute(lap))  # images can't have negative numbers
cv.imshow("Laplacian", lap)

# Sobel
# cumputes differences in one axis
sobelx = cv.Sobel(gray, cv.CV_64F, 1, 0)
cv.imshow("Sobel X", sobelx)
sobely = cv.Sobel(gray, cv.CV_64F, 0, 1)
cv.imshow("Sobel Y", sobely)

combined_sobel = cv.bitwise_or(sobelx, sobely)
cv.imshow("Combined Sobel", combined_sobel)

canny = cv.Canny(gray, 150, 175)
cv.imshow("Canny", canny)

cv.waitKey(0)
