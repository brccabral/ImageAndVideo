import cv2 as cv

# img = cv.imread('Resources/Photos/cat.jpg')

# img = cv.imread('Resources/Photos/cat_large.jpg')

# cv.imshow('Cat', img)

# cv.waitKey(0)

# Reading Videos
# capture = cv.VideoCapture(0) # 0 param is webcam id
capture = cv.VideoCapture("Resources/Videos/dog.mp4")

while True:
    isTrue, frame = capture.read()
    if not isTrue:
        break
    cv.imshow("Video", frame)

    if cv.waitKey(20) & 0xFF == ord("d"):
        break

capture.release()
cv.destroyAllWindows()
