import os
import cv2 as cv
import numpy as np

p = []
for i in os.listdir(r'./Resources/Faces/train'):
    p.append(i)
print(p)