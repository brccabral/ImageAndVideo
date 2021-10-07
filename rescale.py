import cv2 as cv
import numpy

def rescaleFrame(frame: numpy.ndarray, scale: float=0.75):
    # Images, Videos and Stream
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    dimensions = (width, height)

    return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)

def changeResolution(width, height):
    # Live video
    capture.set(3,width)
    capture.set(4,height)

img = cv.imread('Resources/Photos/cat_large.jpg')
img_resized = rescaleFrame(img)
cv.imshow('Cat', img)
cv.imshow('Cat Resized', img_resized)

cv.waitKey(0)

capture = cv.VideoCapture('Resources/Videos/dog.mp4')

isTrue = True
while isTrue:
    isTrue, frame = capture.read()
    if not isTrue:
        break
    frame_resized = rescaleFrame(frame)
    cv.imshow('Video', frame)
    cv.imshow('Video Resized', frame_resized)

    if cv.waitKey(20) & 0xFF==ord('d'):
        break

capture.release()
cv.destroyAllWindows()