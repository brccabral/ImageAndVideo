# %%
import os
import caer
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

# Create training data
train = caer.preprocess_from_dir(DIR=char_path, classes=characters, channels=channels, IMG_SIZE=IMG_SIZE, isShuffle=True, verbose=True)

# %%
len(train)
# %%
import matplotlib.pyplot as plt
plt.figure(figsize=(30,30))
plt.imshow(train[0][0], cmap='gray')
plt.show()
# %%
featureSet, labels = caer.sep_train(train, IMG_SIZE=IMG_SIZE)

# %%
from tensorflow.keras.utils import to_categorical
# Normalize the featureSet ==> (0,1)
featureSet = caer.normalize(featureSet)
labels = to_categorical(labels, len(characters))
# %%
# 20% test
x_train, x_val, y_train, y_val = caer.train_val_split(featureSet, labels, val_ratio=.2)
# %%
del train
del featureSet
del labels
gc.collect()

# %%
BATCH_SIZE = 32
EPOCHS = 10
# Image data generator
datagen = canaro.generators.imageDataGenerator()
train_gen = datagen.flow(x_train, y_train, batch_size=BATCH_SIZE)
# %%
# Creating the model
model = canaro.models.createSimpsonsModel(IMG_SIZE=IMG_SIZE, channels=channels, output_dim=len(characters),
                                          loss='binary_crossentropy', decay=1e-6, learning_rate=0.001,
                                          momentum=0.9, nesterov=True)

# %%
model.summary()

# %%
from tensorflow.keras.callbacks import LearningRateScheduler
callbacks_list = [LearningRateScheduler(canaro.lr_schedule)]

# %%
import scipy
# %%
training = model.fit(train_gen,
                    steps_per_epoch=len(x_train)//BATCH_SIZE,
                    epochs=EPOCHS,
                    validation_data=(x_val, y_val),
                    validation_steps=len(y_val)//BATCH_SIZE,
                    callbacks=callbacks_list)

# %%
