# Introduction
This script implements several Convolutional Neural Network (VGG16, ResNet50) to perform the classification of 2D images. The framework used is Keras by Tensorflow.

The type of data that we are going to manipulate consists in:

    - a jpeg image with 3 channels (RGB)

You can find useful information by reading the official tensorflow tutorials:

    https://www.tensorflow.org/tutorials/keras/classification

The directories that contain the images of the dataset must be organized in the following hierarchy:

```sh
    dataset/
        |
        |- train/
        |       |
        |       |- label0.0.jpg
        |       |- label0.1.jpg
        |       |- label0.2.jpg
        |       | ...      
        |       |- label1.0.jpg
        |       |- label1.1.jpg
        |       |- label1.2.jpg        
        |       | ...
        |       |- label2.0.jpg
        |       |- label2.1.jpg
        |       |- label2.2.jpg
        |       | ...        
        |
        |- test/
               |
               |- label0.t0.jpg
               |- label0.t1.jpg
               |- label0.t2.jpg
               | ...      
               |- label1.t0.jpg
               |- label1.t1.jpg
               |- label1.t2.jpg        
               | ...
               |- label2.t0.jpg
               |- label2.t1.jpg
               |- label2.t2.jpg
               | ...        

```