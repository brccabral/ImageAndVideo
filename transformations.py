import cv2 as cv
import numpy as np

img = cv.imread('Resources/Photos/park.jpg')
cv.imshow('Park', img)

# translation
def translate(img: np.ndarray, x, y):
    transMatrix = np.float32(([1,0,x],[0,1,y]))
    dimensions = (img.shape[1], img.shape[0])
    return cv.warpAffine(img, transMatrix, dimensions)

translated = translate(img, 100, 150)
cv.imshow('Translated Rigth Down', translated)

translated = translate(img, -100, -150)
cv.imshow('Translated Left Up', translated)

cv.waitKey(0)