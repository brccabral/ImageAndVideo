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

def create_train():
    for person in people:
        path = os.path.join(DIR, person)
        label = people.index(person)

        for img in os.listdir(path):
            img_path = os.path.join(path, img)
            print(img_path)
            img_array = cv.imread(img_path)

create_train()
