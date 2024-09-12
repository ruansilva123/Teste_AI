# ========== Imports ==========
import pathlib
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import PIL
import random
from glob import glob
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.image as img
import warnings

import keras
# from tensorflow import keras
from keras.api import layers
# from tensorflow.keras import layers
from keras.api.models import Sequential
# from tensorflow.keras.models import Sequential
from keras.api.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
# from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.api.optimizers import Adam
# from tensorflow.keras.optimizers import Adam 
from keras.api.losses import SparseCategoricalCrossentropy
# from tensorflow.keras.losses import SparseCategoricalCrossentropy


# ========== Identify ==========
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


gpus = tf.config.experimental.list_physical_devices('GPU')
print(gpus)
try:
    tf.config.experimental.set_memory_growth = True
except Exception as ex:
    print(ex)


# ========== Path images ==========
data_dir_train = pathlib.Path("Skin cancer ISIC The International Skin Imaging Collaboration/Train/")
data_dir_test = pathlib.Path("Skin cancer ISIC The International Skin Imaging Collaboration/Test/")


# ========== Variables ==========
batch_size = 32
img_height = 180
img_width = 180
rnd_seed = 123
random.seed(rnd_seed)


# ========== Identify ==========
train_ds = keras.preprocessing.image_dataset_from_directory(
  data_dir_train,
  seed=123,
  validation_split = 0.2,
  subset = 'training',
  image_size=(img_height, img_width),
  batch_size=batch_size)


# ========== Identify ==========
val_ds = keras.preprocessing.image_dataset_from_directory(
  data_dir_test,
  seed=123,
  validation_split = 0.2,
  subset = 'validation',
  image_size=(img_height, img_width),
  batch_size=batch_size)


# ========== Identify ==========
print(f"\n\nClasses: {train_ds.class_names}\n\n")


# ========== Identify ==========
num_classes = 9
model = Sequential([layers.Rescaling(1.0/255,input_shape=(img_height,img_width,3))])

model.add(Conv2D(32, 3,padding="same",activation='relu'))
model.add(MaxPool2D())

model.add(Conv2D(64, 3,padding="same",activation='relu'))
model.add(MaxPool2D())

model.add(Conv2D(128, 3,padding="same",activation='relu'))
model.add(MaxPool2D())
model.add(Dropout(0.15))

model.add(Conv2D(256, 3,padding="same",activation='relu'))
model.add(MaxPool2D())
model.add(Dropout(0.20))

model.add(Conv2D(512, 3,padding="same",activation='relu'))
model.add(MaxPool2D())
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(1024,activation="relu"))
model.add(Dense(units=num_classes, activation= 'softmax'))


# ========== Identify ==========
opt = Adam(learning_rate=0.001)
model.compile(optimizer= opt,
              loss = SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])


# ========== Identify ==========
epochs = 25
history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs
)


# ========== Identify ==========
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()


# ========== Identify ==========
top_model_weights_path = 'Sizes/cnn_fc_model.weights.h5'
model.save_weights(top_model_weights_path)


# ========== Identify ==========
# (eval_loss, eval_accuracy) = model.evaluate(test_ds, batch_size=batch_size, \
#                                             verbose=1)


# ========== Identify ==========
# print("[INFO] accuracy: {:.2f}%".format(eval_accuracy * 100)) 
# print("[INFO] Loss: {}".format(eval_loss)) 