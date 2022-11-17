"""
    Neural Network implementation for image classification

    Copyright (c) 2022 Giansalvo Gusinu

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
from keras.applications.mobilenet_v2 import MobileNetV2
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense, GlobalAveragePooling2D, Dropout, BatchNormalization, Input
from keras.models import Sequential, Model

INPUT_SHAPE =(224,224,3)

base_model = MobileNetV2(
    weights = "imagenet", 
    input_shape = INPUT_SHAPE,
    include_top = False
)


def create_MobileNetV2_keras(num_classes=2, freeze_base = False):
    if freeze_base:
        for layer in base_model.layers:
            layer.trainable = False

    model= Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(100, activation='relu'),
        Dropout(0.4),
        Dense(64, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    return model

def main():
    model = create_MobileNetV2_keras(2)
    model.compile(loss = "categorical_crossentropy", optimizer = "adam", metrics = "accuracy")
    model.summary()

    return

if __name__ == '__main__':
    main()