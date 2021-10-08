import os
import cv2 as cv
import numpy as np

DIR = r'./Resources/Faces/train'

people = []
for i in os.listdir(DIR):
    people.append(i)
print(people)

features = []
labels = []

haar_cascade: cv.CascadeClassifier = cv.CascadeClassifier('haarcascade_frontalface_default.xml')

def create_train():
    for person in people:
        path = os.path.join(DIR, person)
        label = people.index(person)

        for img in os.listdir(path):
            img_path = os.path.join(path, img)
            # print(img_path)
            img_array = cv.imread(img_path)

            gray = cv.cvtColor(img_array, cv.COLOR_BGR2GRAY)
            faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

            for (x,y,w,h) in faces_rect:
                faces_roi = gray[y:y+h, x:x+w]
                features.append(faces_roi)
                labels.append(label)
                #cv.imshow(f'{person} {img}', faces_roi)
        #break

create_train()

print(f'Length features {len(features)}')
print(f'Length labels {len(labels)}')

features = np.array(features, dtype=object)
labels = np.array(labels)

np.save('features.npy', features)
np.save('labels.npy', labels)

face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.train(features, labels)
face_recognizer.save('face_trained.yml')

print('Training done ------------')

# cv.waitKey(0)
