import cv2 as cv

img = cv.imread('Resources/Photos/park.jpg')
cv.imshow('Park', img)

# convert to grayscale
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('Gray', gray)

# blur
blur = cv.GaussianBlur(img, (3,3), cv.BORDER_DEFAULT)
cv.imshow('Blur', blur)
blur = cv.GaussianBlur(img, (7,7), cv.BORDER_DEFAULT)
cv.imshow('More Blur', blur)

# Edge Cascade
canny = cv.Canny(img, 125, 175)
cv.imshow('Canny Edges', canny)

# reduce the amount of edges by passing blur images
canny = cv.Canny(blur, 125, 175)
cv.imshow('Less Edges', canny)

# dilating the image
dilated = cv.dilate(canny, (7,7), iterations=3)
cv.imshow('Dilated', dilated)

# Eroding
eroded = cv.erode(dilated, (3,3), iterations=3)
cv.imshow('Eroded', eroded)

eroded = cv.erode(dilated, (7,7), iterations=3) # get back almost the same as Less Edges
cv.imshow('Eroded', eroded)

# resize
resized = cv.resize(img, (500,500)) # don't keep aspect ratio
cv.imshow('Resized', resized)

resized = cv.resize(img, (500,500), interpolation=cv.INTER_CUBIC) # better resolution if shape is different from original
cv.imshow('Resized INTER_CUBIC', resized)

# Cropping
cropped = img[50:200, 200:400]
cv.imshow('Cropped', cropped)

cv.waitKey(0)