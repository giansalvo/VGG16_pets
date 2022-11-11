"""
    Neural Network implementation for image classification

    Copyright (c) 2022 Giansalvo Gusinu
    Copyright (c) Sarit Rath 

    Code adapted from following articles/repositories:
    https://www.kaggle.com/code/saritrath/cats-vs-dogs-vgg-16

    Permission is hereby granted, free of charge, to any person obtaining a 
    copy of this software and associated documentation files (the "Software"),
    to deal in the Software without restriction, including without limitation
    the rights to use, copy, modify, merge, publish, distribute, sublicense,
    and/or sell copies of the Software, and to permit persons to whom the
    Software is furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in
    all copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
    THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
    FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
    DEALINGS IN THE SOFTWARE.
"""
import argparse
import logging
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import sys
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, classification_report 
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import load_img
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from keras.applications.vgg16 import preprocess_input
from keras.models import Model
from tensorflow.keras.preprocessing.image import img_to_array

from model_keras_VGG16 import create_VGG16_keras

SEED = 666

BATCH_SIZE_DEFAULT = 128
NUM_CLASSES_DEFAULT = 2

DATASET_TRAIN_SUBDIR = "train"
FILES_WEIGHTS = "weights.hdf5"

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
ACTION_EVALUATE = "evaluate"


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
                "         $python %(prog)s train -dr dataset_root_folder -w weights_file [-b batch_size] [-c num_classes]\n"
                "\n"
                "       Make prediction for an image\n"
                "         $python %(prog)s predict -i input_image -w weights_file [-c num_classes]\n"
                "\n"
                "       Evaluate the network and compute confusion matrix and performance indexes\n"
                "         $python %(prog)s evaluate -dr dataset_root_folder -w weights_file\n"
                "\n",
        formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--version', action='version', version='%(prog)s v.' + PROGRAM_VERSION)
    parser.add_argument("action", help="The action to be performed.")
    parser.add_argument('--check', dest='check', default=False, action='store_true',
                    help="Display some images from dataset before training to check that dataset is ok.")
    parser.add_argument('-dr', '--dataset_root_dir', required=False, help="The root directory of the dataset.")
    parser.add_argument("-b", "--batch_size", required=False, default=BATCH_SIZE_DEFAULT, type=int, help="the number of samples that are passed to the network at once during the training")
    parser.add_argument('-c', "--num_classes", required=False, default=NUM_CLASSES_DEFAULT, type=int,
                        help="The number of possible classes for an images.")
    parser.add_argument("-i", "--input_image", required=False, help="The input file to be classified.")
    parser.add_argument("-w", "--weigths_file", required=False, default=FILES_WEIGHTS,
                        help="The file where the network weights will be saved. It must be compatible with the network model chosen.")
                     
    args = parser.parse_args()
    
    action = args.action
    check = args.check
    dataset_root_dir = args.dataset_root_dir
    batch_size = args.batch_size
    num_classes = args.num_classes
    input_image = args.input_image
    weights_fname = args.weigths_file

    if action == ACTION_TRAIN or action == ACTION_EVALUATE:
        if dataset_root_dir is None:
            raise ValueError('ERROR: parameter dataset_root_dir not specified. Check syntax with --help')
        files_train_path = os.path.join(dataset_root_dir, DATASET_TRAIN_SUBDIR)

        train_df = pd.DataFrame({"file": os.listdir(files_train_path)})
        train_df["label"] = train_df["file"].apply(lambda x: x.split(".")[0])
        print(train_df.head())

        # test_df = pd.DataFrame({"file": os.listdir(FILES_TEST)})
        # print(test_df.head())

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
            fn = os.path.join(files_train_path, train_df["file"][i])
            image = load_img(fn)
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
            fn = os.path.join(files_train_path, train_df.query("label == 'dog'").file.values[i])
            image = load_img(fn)
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
            fn = os.path.join(files_train_path, train_df.query("label == 'cat'").file.values[i])
            image = load_img(fn)
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
        logger.debug("train_data.shape={}".format(train_data.shape))
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
            directory = files_train_path,
            x_col = "file",
            y_col = "label",
            class_mode = "categorical",
            target_size = (224, 224),
            batch_size = batch_size,
            seed = SEED,
        )

        val_generator = val_datagen.flow_from_dataframe(
            dataframe = val_data,
            directory = files_train_path,
            x_col = "file",
            y_col = "label",
            class_mode = "categorical",
            target_size = (224, 224),
            batch_size = batch_size,
            seed = SEED,
            shuffle = False
        )

    if action == ACTION_TRAIN:
        model = create_VGG16_keras(num_classes, freeze_base=True)
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
            filepath = weights_fname,
            verbose = 1,
            save_best_only = True, 
            save_weights_only = True
        )

        validation_samples_num = val_data.shape[0]
        logger.debug("validation_samples_num={}".format(validation_samples_num))
        validation_steps = val_data.shape[0] // batch_size
        logger.debug("validation_steps={}".format(validation_steps))
        steps_per_epoch = train_data.shape[0] // batch_size
        logger.debug("steps_per_epoch={}".format(steps_per_epoch))
        if validation_steps == 0:
            raise ValueError('ERROR: validation_steps is null. Reduce number of batch_size. Check syntax with --help.')
        history = model.fit(
            train_generator,
            epochs = 10, 
            validation_data = val_generator,
            validation_steps = validation_steps,
            steps_per_epoch = steps_per_epoch,
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

    elif action == ACTION_PREDICT:
        if input_image is None:
            raise("ERROR: input_image must be provided. Check syntax with --help.")
        model = create_VGG16_keras(num_classes)
        model.load_weights(weights_fname)

        # predict
        image = load_img(input_image, target_size=(224, 224))
        image = img_to_array(image)
        # reshape data for the model
        image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
        pred = model.predict(image)
        im_class = np.argmax(pred)
        print("Image of classe: {}".format(im_class))

    elif action == ACTION_EVALUATE:
        model = create_VGG16_keras(num_classes)
        model.load_weights(weights_fname)

        # predict all
        val_pred = model.predict(val_generator, steps = np.ceil(val_data.shape[0] / batch_size))

        # Compute confusion matrix
        val_data.loc[:, "val_pred"] = np.argmax(val_pred, axis = 1)
        labels = dict((v, k) for k, v in val_generator.class_indices.items())
        labels_names = []
        num_classes = len(labels)
        for i in range(num_classes):
            labels_names.append(labels[i])
        val_data.loc[:, "val_pred"] = val_data.loc[:, "val_pred"].map(labels)
        fig, ax = plt.subplots(figsize = (9, 6))
        cm = confusion_matrix(val_data["label"], val_data["val_pred"])

        # Compute performance indexes
        TP = np.diag(cm)
        FP = np.sum(cm, axis=0) - TP
        FN = np.sum(cm, axis=1) - TP
        TN = []
        for i in range(num_classes):
            temp = np.delete(cm, i, 0)    # delete ith row
            temp = np.delete(temp, i, 1)  # delete ith column
            TN.append(sum(sum(temp)))
        # Overall accuracy
        ACC = (TP+TN)/(TP+FP+FN+TN)
        # # sanity check
        # for i in range(num_classes):
        #     print(TP[i] + FP[i] + FN[i] + TN[i])
        precision = TP/(TP+FP)
        recall = TP/(TP+FN)
        specificity = TN/(TN+FP)
        print("classes: " + str(labels_names))
        print("accuracy: " + str(ACC))
        print("precision: " + str(precision))
        print("recall: " + str(recall))
        print("specificity: " + str(specificity))                
        # Compute and save confusion Matrix diagram
        disp = ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = labels_names)
        disp.plot(cmap = plt.cm.Blues, ax = ax)
        ax.set_title("Validation Set")
        logger.debug("Saving Confusion Matrix to file {}...".format(FILE_CM))
        plt.savefig(FILE_CM)
        plt.close()

    logger.debug("End of program.\n")
    return


if __name__ == '__main__':
    main()