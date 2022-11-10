# adapted from Sarit Rath https://www.kaggle.com/code/saritrath/cats-vs-dogs-vgg-16
import argparse
import logging
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import sys
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

SEED = 666

BATCH_SIZE = 128
INPUT_SHAPE =(224,224,3)

TRAIN_PATH = "./train.zip"
TEST_PATH = "./test1.zip"

FILES = "./Images/"
FILES_TRAIN = "./Images/train"
FILES_TEST = "./Images/test1"
FILES_WEIGHTS = "catdog_vgg16.hdf5"

# Output performance file names
FILE_AUGMENT = "./perf_augment.png"
FILE_CM = "./perf_cm.png"
FILE_DISTRIBUTION = "./perf_distrib.png"
FILE_SAMPLES = "./perf_samples.png"
FILE_SAMPLES_DOGS = "./perf_samples_dogs.png"
FILE_SAMPLES_CATS = "./perf_samples_cats.png"

# COPYRIGHT NOTICE AND PROGRAM VERSION
COPYRIGHT_NOTICE = "Copyright (C) 2022 Giansalvo Gusinu"
PROGRAM_VERSION = "1.0"

ACTION_TRAIN = "train"
ACTION_PREDICT = "predict"

def extract_files():
    with zipfile.ZipFile(TRAIN_PATH, 'r') as zipp:
        zipp.extractall(FILES)
        
    with zipfile.ZipFile(TEST_PATH, 'r') as zipp:
        zipp.extractall(FILES)

base_model = VGG16(
    weights = "imagenet", 
    input_shape = INPUT_SHAPE,
    include_top = False
)


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

#########################################
# Main
#########################################
def main():
    tf.random.set_seed(SEED)
    np.random.seed(SEED)
    os.environ["PYTHONHASHSEED"] = str(SEED)                      
    random.seed(SEED)

    # create logger
    logger = logging.getLogger('gians')
    logger.setLevel(logging.DEBUG)
    # create console handler and set level to debug
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    # create formatter
    formatter = logging.Formatter('%(asctime)s-%(name)s-%(levelname)s:%(message)s')
    # add formatter to ch
    ch.setFormatter(formatter)
    # add ch to logger
    logger.addHandler(ch)
    logger.info("Starting")
    
    print("Program ver.: " + PROGRAM_VERSION)
    print(COPYRIGHT_NOTICE)
    executable = os.path.realpath(sys.executable)
    logger.info("Python ver.: " + sys.version)
    logger.info("Python executable: " + str(executable))
    logger.info("Tensorflow ver. " + str(tf.__version__))
    # Print invocation command line
    cmd_line = ""
    narg = len(sys.argv)
    for x in range(narg):
        cmd_line += " " + sys.argv[x]
    logger.debug("Invocation command: " + cmd_line)        

    parser = argparse.ArgumentParser(
        description=COPYRIGHT_NOTICE,
        epilog = "Examples:\n"
                "       Train the network\n"
                "         $python %(prog)s train\n"
                "\n"
                "       Make predictions and compute confusion matrix\n"
                "         $python %(prog)s predict\n"
                "\n",
        formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--version', action='version', version='%(prog)s v.' + PROGRAM_VERSION)
    parser.add_argument("action", help="The action to be performed.")
    parser.add_argument('--check', dest='check', default=False, action='store_true',
                    help="Display some images from dataset before training to check that dataset is ok.")

    args = parser.parse_args()
    
    action = args.action
    check = args.check

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
    logger.debug("Saving Distribution Diagram to file {}...".format(FILE_DISTRIBUTION))
    plt.savefig(FILE_DISTRIBUTION)
    plt.close()


    fig = plt.figure(1, figsize = (8, 8))
    fig.suptitle("Training Set Images (Sample)")
    for i in range(25):
        plt.subplot(5, 5, i + 1)
        image = load_img(FILES + "train/" + train_df["file"][i])
        plt.imshow(image)
        plt.axis("off")
    plt.tight_layout()
    logger.debug("Saving samples to file {}...".format(FILE_SAMPLES))
    plt.savefig(FILE_SAMPLES)
    plt.close()


    fig = plt.figure(1, figsize = (8, 8))
    fig.suptitle("Sample Dog images from Training Set")
    for i in range(25):
        plt.subplot(5, 5, i + 1)
        image = load_img(FILES + "train/" + train_df.query("label == 'dog'").file.values[i])
        plt.imshow(image)
        plt.axis("off")
    plt.tight_layout()
    logger.debug("Saving Sample Dog images from Training Set to file {}...".format(FILE_SAMPLES_DOGS))
    plt.savefig(FILE_SAMPLES_DOGS)
    plt.close()

    fig = plt.figure(1, figsize = (8, 8))
    fig.suptitle("Sample Cat images from Training Set")
    for i in range(25):
        plt.subplot(5, 5, i + 1)
        image = load_img(FILES + "train/" + train_df.query("label == 'cat'").file.values[i])
        plt.imshow(image)
        plt.axis("off")
    plt.tight_layout()
    logger.debug("Saving Sample Cats images from Training Set to file {}...".format(FILE_SAMPLES_CATS))
    plt.savefig(FILE_SAMPLES_CATS)
    plt.close()


    train_data, val_data = train_test_split(train_df, 
                                            test_size = 0.2, 
                                            stratify = train_df["label"], 
                                            random_state = SEED)
    # datagen = ImageDataGenerator(
    #     rotation_range = 30, 
    #     width_shift_range = 0.1,
    #     height_shift_range = 0.1, 
    #     brightness_range = (0.5, 1), 
    #     zoom_range = 0.2,
    #     horizontal_flip = True, 
    #     rescale = 1./255,
    # )
    # sample_df = train_data.sample(1)
    # sample_generator = datagen.flow_from_dataframe(
    #     dataframe = sample_df,
    #     directory = FILES + "train/",
    #     x_col = "file",
    #     y_col = "label",
    #     class_mode = "categorical",
    #     target_size = (224, 224),
    #     seed = SEED
    # )

    # fig = plt.figure(figsize = (14, 8))
    # fig.suptitle("Augmentation techniques")
    # for i in range(50):
    #     plt.subplot(5, 10, i + 1)
    #     for X, y in sample_generator:
    #         plt.imshow(X[0])
    #         plt.axis("off")
    #         break
    # plt.tight_layout()
    # if check:
    #     plt.show()
    # else:
    #     logger.debug("Saving augmentation samples to file {}...".format(FILE_AUGMENT))
    #     plt.savefig(FILE_AUGMENT)
    #     plt.close()

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
        batch_size = BATCH_SIZE,
        seed = SEED,
    )

    val_generator = val_datagen.flow_from_dataframe(
        dataframe = val_data,
        directory = FILES + "train/",
        x_col = "file",
        y_col = "label",
        class_mode = "categorical",
        target_size = (224, 224),
        batch_size = BATCH_SIZE,
        seed = SEED,
        shuffle = False
    )

    if action == ACTION_TRAIN:
        for layer in base_model.layers:
            layer.trainable = False

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
            validation_steps = val_data.shape[0] // BATCH_SIZE,
            steps_per_epoch = train_data.shape[0] // BATCH_SIZE,
            callbacks = [reduce_lr, early_stopping, checkpoint]
        )

        fig, axes = plt.subplots(1, 2, figsize = (12, 4))
        sns.lineplot(x = range(len(history.history["loss"])), y = history.history["loss"], ax = axes[0], label = "Training Loss")
        sns.lineplot(x = range(len(history.history["loss"])), y = history.history["val_loss"], ax = axes[0], label = "Validation Loss")
        sns.lineplot(x = range(len(history.history["accuracy"])), y = history.history["accuracy"], ax = axes[1], label = "Training Accuracy")
        sns.lineplot(x = range(len(history.history["accuracy"])), y = history.history["val_accuracy"], ax = axes[1], label = "Validation Accuracy")
        axes[0].set_title("Loss"); axes[1].set_title("Accuracy")
        sns.despine()
        FILE_ACCURACY = "./perf_accuracy.png"
        logger.debug("Saving Loss and Accuracy functions to file {}...".format(FILE_ACCURACY))
        plt.savefig(FILE_ACCURACY)
        plt.close()

        tf.keras.backend.clear_session()
        
    elif action == ACTION_PREDICT:
        model = vgg16_pretrained()
        model.load_weights(FILES_WEIGHTS)

        val_pred = model.predict(val_generator, steps = np.ceil(val_data.shape[0] / BATCH_SIZE))
        val_data.loc[:, "val_pred"] = np.argmax(val_pred, axis = 1)

        labels = dict((v, k) for k, v in val_generator.class_indices.items())

        val_data.loc[:, "val_pred"] = val_data.loc[:, "val_pred"].map(labels)


        fig, ax = plt.subplots(figsize = (9, 6))

        cm = confusion_matrix(val_data["label"], val_data["val_pred"])

        disp = ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = ["cat", "dog"])
        disp.plot(cmap = plt.cm.Blues, ax = ax)
        ax.set_title("Validation Set")
        logger.debug("Saving Confusion Matrix to file {}...".format(FILE_CM))
        plt.savefig(FILE_CM)
        plt.close()

    logger.debug("End of program.\n")
    return


if __name__ == '__main__':
    main()