# adapted from Sarit Rath https://www.kaggle.com/code/saritrath/cats-vs-dogs-vgg-16
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import random
import zipfile
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, classification_report 
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import load_img
#from keras.preprocessing.image import load_img, ImageDataGenerator
from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense, GlobalAveragePooling2D, Dropout, BatchNormalization, Input
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.applications.vgg16 import VGG16
from keras.models import Model

batch_size = 128
seed = 666
tf.random.set_seed(seed)
np.random.seed(seed)
os.environ["PYTHONHASHSEED"] = str(seed)                      
random.seed(seed)

TRAIN_PATH = "./train.zip"
TEST_PATH = "./test1.zip"

FILES = "./Images/"
FILES_TRAIN = "./Images/train"
FILES_TEST = "./Images/test1"
FILES_WEIGHTS = "catdog_vgg16.hdf5"

def extract_files():
    with zipfile.ZipFile(TRAIN_PATH, 'r') as zipp:
        zipp.extractall(FILES)
        
    with zipfile.ZipFile(TEST_PATH, 'r') as zipp:
        zipp.extractall(FILES)


train_df = pd.DataFrame({"file": os.listdir(FILES_TRAIN)})
train_df["label"] = train_df["file"].apply(lambda x: x.split(".")[0])
print(train_df.head())

test_df = pd.DataFrame({"file": os.listdir(FILES_TEST)})
print(test_df.head())


fig, ax = plt.subplots(figsize = (6, 6), facecolor = "#e5e5e5")
ax.set_facecolor("#e5e5e5")

sns.countplot(x = "label", data = train_df, ax = ax)

ax.set_title("Distribution of Class Labels")
sns.despine()
plt.show()



fig = plt.figure(1, figsize = (8, 8))
fig.suptitle("Training Set Images (Sample)")
for i in range(25):
    plt.subplot(5, 5, i + 1)
    image = load_img(FILES + "train/" + train_df["file"][i])
    plt.imshow(image)
    plt.axis("off")
plt.tight_layout()
plt.show()


fig = plt.figure(1, figsize = (8, 8))
fig.suptitle("Sample Dog images from Training Set")
for i in range(25):
    plt.subplot(5, 5, i + 1)
    image = load_img(FILES + "train/" + train_df.query("label == 'dog'").file.values[i])
    plt.imshow(image)
    plt.axis("off")
plt.tight_layout()
plt.show()



fig = plt.figure(1, figsize = (8, 8))
fig.suptitle("Sample Cat images from Training Set")
for i in range(25):
    plt.subplot(5, 5, i + 1)
    image = load_img(FILES + "train/" + train_df.query("label == 'cat'").file.values[i])
    plt.imshow(image)
    plt.axis("off")
plt.tight_layout()
plt.show()


train_data, val_data = train_test_split(train_df, 
                                        test_size = 0.2, 
                                        stratify = train_df["label"], 
                                        random_state = 666)
datagen = ImageDataGenerator(
    rotation_range = 30, 
    width_shift_range = 0.1,
    height_shift_range = 0.1, 
    brightness_range = (0.5, 1), 
    zoom_range = 0.2,
    horizontal_flip = True, 
    rescale = 1./255,
)
sample_df = train_data.sample(1)
sample_generator = datagen.flow_from_dataframe(
    dataframe = sample_df,
    directory = FILES + "train/",
    x_col = "file",
    y_col = "label",
    class_mode = "categorical",
    target_size = (224, 224),
    seed = 666
)
fig = plt.figure(figsize = (14, 8))
fig.suptitle("Augmentation techniques")
for i in range(50):
    plt.subplot(5, 10, i + 1)
    for X, y in sample_generator:
        plt.imshow(X[0])
        plt.axis("off")
        break
plt.tight_layout()
plt.show()

train_datagen = ImageDataGenerator(
    rotation_range = 15, 
#     width_shift_range = 0.1,
#     height_shift_range = 0.1, 
#     brightness_range = (0.5, 1), 
#     zoom_range = 0.1,
    horizontal_flip = True,
    preprocessing_function = preprocess_input
)

val_datagen = ImageDataGenerator(preprocessing_function = preprocess_input)
train_generator = train_datagen.flow_from_dataframe(
    dataframe = train_data,
    directory = FILES + "train/",
    x_col = "file",
    y_col = "label",
    class_mode = "categorical",
    target_size = (224, 224),
    batch_size = batch_size,
    seed = 666,
)

val_generator = val_datagen.flow_from_dataframe(
    dataframe = val_data,
    directory = FILES + "train/",
    x_col = "file",
    y_col = "label",
    class_mode = "categorical",
    target_size = (224, 224),
    batch_size = batch_size,
    seed = 666,
    shuffle = False
)


input_shape=(224,224,3)
batch_size= 128

base_model = VGG16(
    weights = "imagenet", 
    input_shape = (224, 224, 3),
    include_top = False
)

for layer in base_model.layers:
    layer.trainable = False
    
def vgg16_pretrained():
    
    model= Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(100,activation='relu'),
        Dropout(0.4),
        Dense(64,activation='relu'),
        Dense(2,activation='softmax')
    ])
    
    return model


tf.keras.backend.clear_session()

model = vgg16_pretrained()
model.compile(loss = "categorical_crossentropy", optimizer = "adam", metrics = "accuracy")
model.summary()

reduce_lr = ReduceLROnPlateau(
    monitor = "val_accuracy", 
    patience = 2,
    verbose = 1, 
    factor = 0.5, 
    min_lr = 0.000000001
)

early_stopping = EarlyStopping(
    monitor = "val_accuracy",
    patience = 5,
    verbose = 1,
    mode = "max",
)

checkpoint = ModelCheckpoint(
    monitor = "val_accuracy",
    filepath = FILES_WEIGHTS,
    verbose = 1,
    save_best_only = True, 
    save_weights_only = True
)


history = model.fit(
    train_generator,
    epochs = 10, 
    validation_data = val_generator,
    validation_steps = val_data.shape[0] // batch_size,
    steps_per_epoch = train_data.shape[0] // batch_size,
    callbacks = [reduce_lr, early_stopping, checkpoint]
)


tf.keras.backend.clear_session()
model = vgg16_pretrained()
model.load_weights(FILES_WEIGHTS)


fig, axes = plt.subplots(1, 2, figsize = (12, 4))
sns.lineplot(x = range(len(history.history["loss"])), y = history.history["loss"], ax = axes[0], label = "Training Loss")
sns.lineplot(x = range(len(history.history["loss"])), y = history.history["val_loss"], ax = axes[0], label = "Validation Loss")
sns.lineplot(x = range(len(history.history["accuracy"])), y = history.history["accuracy"], ax = axes[1], label = "Training Accuracy")
sns.lineplot(x = range(len(history.history["accuracy"])), y = history.history["val_accuracy"], ax = axes[1], label = "Validation Accuracy")
axes[0].set_title("Loss"); axes[1].set_title("Accuracy")
sns.despine()
plt.show()

val_pred = model.predict(val_generator, steps = np.ceil(val_data.shape[0] / batch_size))
val_data.loc[:, "val_pred"] = np.argmax(val_pred, axis = 1)

labels = dict((v, k) for k, v in val_generator.class_indices.items())

val_data.loc[:, "val_pred"] = val_data.loc[:, "val_pred"].map(labels)


fig, ax = plt.subplots(figsize = (9, 6))

cm = confusion_matrix(val_data["label"], val_data["val_pred"])

disp = ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = ["cat", "dog"])
disp.plot(cmap = plt.cm.Blues, ax = ax)

ax.set_title("Validation Set")
plt.show()


