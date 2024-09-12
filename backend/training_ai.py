# ========== Imports ==========
import pathlib
import keras
from keras.api import layers
from keras.api.models import Sequential
from keras.api.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.api.optimizers import Adam
from keras.api.losses import SparseCategoricalCrossentropy


# ========== Path images ==========
data_dir_train = pathlib.Path("Skin cancer ISIC The International Skin Imaging Collaboration/Train/")
data_dir_test = pathlib.Path("Skin cancer ISIC The International Skin Imaging Collaboration/Test/")


# ========== Variables ==========
batch_size = 32
img_height = 180
img_width = 180


# ========== Load and init datas ==========
train_ds = keras.preprocessing.image_dataset_from_directory(
  data_dir_train,
  seed=123,
  validation_split = 0.2,
  subset = 'training',
  image_size=(img_height, img_width),
  batch_size=batch_size)


# ========== Load and init datas ==========
val_ds = keras.preprocessing.image_dataset_from_directory(
  data_dir_test,
  seed=123,
  validation_split = 0.2,
  subset = 'validation',
  image_size=(img_height, img_width),
  batch_size=batch_size)


# ========== Create model ==========
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
model.add(Dense(units=9, activation= 'softmax'))


# ========== Compile model ==========
model.compile(optimizer= Adam(learning_rate=0.001),
              loss = SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])


# ========== Trein model ==========
history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=25
)


# ========== Save model ==========
model.save_weights('Sizes/cnn_fc_model.weights.h5')