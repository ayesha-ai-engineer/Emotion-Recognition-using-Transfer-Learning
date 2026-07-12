import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2

IMG_SIZE = 48
BATCH = 64
EPOCHS = 55

# from google.colab import drive   #for storing the trained model on the drive
# drive.mount('/content/drive')

# from google.colab import files   #using this for uploading the dataset zip file
# uploaded = files.upload()

# !unzip /content/dataset.zip -d /content/   #unzipping the uploaded zip file of dataset

DATASET_DIR = "/content/dataset"   #path to ur dataset

train_dir = os.path.join(DATASET_DIR, "train")
val_dir = os.path.join(DATASET_DIR, "test")

# data augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_directory(
    train_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH,
    class_mode="categorical",
    color_mode="rgb"
)

val_gen = val_datagen.flow_from_directory(
    val_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH,
    class_mode="categorical",
    color_mode="rgb"
)

NUM_CLASSES = len(train_gen.class_indices)
print("Classes:", train_gen.class_indices)

# importing the base model MobileNetV2 and making it to fit for the 7 types of emotions
base = MobileNetV2(
    include_top=False,
    weights="imagenet",
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)

base.trainable = False  # freeze base model

x = base.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x)
x = Dense(256, activation="relu")(x)
x = Dropout(0.2)(x)
output = Dense(NUM_CLASSES, activation="softmax")(x)

model = Model(inputs=base.input, outputs=output)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# training the model
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS
)

# saving it before fine tuning
folder_path = '/content/drive/MyDrive/AI_ML'   #path in drive where u wanna store the model
model.save(os.path.join(folder_path, 'transfer_learning_model_with_mobile_net_v2_no_fine_tuning.keras'))

# fine tuning by
for layer in base.layers[-30:]:
    layer.trainable = True

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

history_ft = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=20
)

#evaluating the model
val_loss, val_acc = model.evaluate(val_gen)
print(f"Validation Accuracy: {val_acc*100:.2f}%")  #before fine tuning

val_loss, val_acc = model.evaluate(val_gen) #after fine tuning
print(f"Validation Accuracy: {val_acc*100:.2f}%")

# saving the model after fine tuning
folder_path = '/content/drive/MyDrive/AI_ML'
model.save(os.path.join(folder_path, 'transfer_learning_model_with_mobile_net_v2_after_fine_tuning.keras'))

model.summary()

# testing on an img from the dataset
import cv2
import numpy as np

img_path="/content/dataset/test/angry/PrivateTest_11296953.jpg"
def predict_emotion(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img, (48, 48))  # FER2013 input size
    img_resized = img_resized / 255.0
    img_input = np.expand_dims(img_resized, axis=0)

    preds = model.predict(img_input)
    class_idx = np.argmax(preds)

    inv_map = {v: k for k, v in train_gen.class_indices.items()}
    return inv_map[class_idx]

# Example
print(predict_emotion("/content/dataset/test/angry/PrivateTest_11296953.jpg"))

