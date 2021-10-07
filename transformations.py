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

# rotation
def rotate(img: np.ndarray, angle, rotPoint=None):
    (height, width) = img.shape[:2]

    if rotPoint is None:
        rotPoint = (width//2,height//2)

    # center, angle, scale
    rotMatrix = cv.getRotationMatrix2D(rotPoint, angle, 1.0)
    dimensions = (width, height)

    return cv.warpAffine(img, rotMatrix, dimensions)

rotated = rotate(img, 45)
cv.imshow('Rotate', rotated)

# resized
resized = cv.resize(img, (500,500), interpolation=cv.INTER_CUBIC) # better resolution if shape is different from original
cv.imshow('Resized INTER_CUBIC', resized)

# flipping
flip = cv.flip(img, 1)
cv.imshow('Flip horizontal', flip)
flip = cv.flip(img, 0)
cv.imshow('Flip vertical', flip)
flip = cv.flip(img, -1)
cv.imshow('Flip both horizontal vertical', flip)


cv.waitKey(0)