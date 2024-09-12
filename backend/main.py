# ========== Imports FastAPI ==========
from fastapi import FastAPI, UploadFile, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware


# ========== Imports libs ==========
import os
import numpy as np
import keras
from keras.api import layers
from keras.api.models import Sequential
from keras.api.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
import tensorflow as tf
from keras.api.optimizers import Adam
from keras.api.losses import SparseCategoricalCrossentropy


# ========== Config FastAPI ==========
app = FastAPI(debug=True)

origins = ["*"]

# origins = [
#     "http://localhost:8000",  
#     "http://127.0.0.1:5500",  
# ]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  
    allow_credentials=True,  
    allow_methods=["*"],  
    allow_headers=["*"],  
)


# ========== Variables ==========
IMG_HEIGHT = 180
IMG_WIDTH = 180
CLASS_NAMES = [
    'actinic keratosis', 'basal cell carcinoma', 'dermatofibroma', 'melanoma', 'nevus', 'pigmented benign keratosis', 
    'seborrheic keratosis', 'squamous cell carcinoma', 'vascular lesion'
]


# ========== AI model ==========
model = Sequential([layers.Rescaling(1.0/255,input_shape=(IMG_HEIGHT,IMG_WIDTH,3))])

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


# ========== Load train datas ==========
model.load_weights("Sizes/cnn_fc_model.weights.h5")


# ========== Compile model ==========
model.compile(
    optimizer= Adam(learning_rate=0.001),
    loss = SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)


# ========== Type of skin cancer ==========
def verify_image(file_name):
    img = keras.api.utils.load_img(f"uploaded_images/{file_name}", target_size=(IMG_HEIGHT, IMG_WIDTH))
    img_array = keras.api.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)

    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])

    predicted_class = CLASS_NAMES[tf.argmax(score)]
    confidence = 100 * np.max(score)

    return predicted_class, confidence


# ========== Function to process image ==========
@app.post("/upload-image/")
async def process_image(file: UploadFile):
    
    # ========== Verify type of file ==========
    if file.content_type not in ["image/jpeg", "image/png", "image/jpg"]:
        return JSONResponse(content={"error" : "Invalid file type!"}, status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE)

    # ========== Save on server ==========
    with open(f"uploaded_images/{file.filename}", "wb") as image:
        content = await file.read()
        image.write(content)
    
    predicted_class, confidence = verify_image(file.filename)

    os.remove(f"uploaded_images/{file.filename}")

    return JSONResponse(content={"predicted_class" : predicted_class, "confidence" : confidence}, status_code=status.HTTP_200_OK)


# ========== Initialize server ==========
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", reload=True)