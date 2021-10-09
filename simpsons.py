# %%
import os
import caer
from caer.io.io import imsave
import canaro
import numpy as np
import cv2 as cv
import gc

# %%
IMG_SIZE = (80,80)
channels = 1
char_path = r'Resources/simpsons_dataset'

# %%
char_dict = {}
for char in os.listdir(char_path):
    char_dict[char] = len(os.listdir(os.path.join(char_path, char)))
char_dict = caer.sort_dict(char_dict, descending=True)
char_dict
# %%
characters = []
count = 0
for i in char_dict:
    characters.append(i[0])
    count += 1
    if count >= 10:
        break
characters
# %%

# %%
