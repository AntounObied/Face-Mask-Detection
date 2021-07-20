"""
@author Antoun Obied

This program generates image training and validation data from a directory,
and trains a CNN to perform classification for mask detection.
"""

import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Flatten, Dense, Dropout, AveragePooling2D, Input
from tensorflow.keras.preprocessing.image import img_to_array, ImageDataGenerator, load_img
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, MobileNetV2
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import sys

# Some parameters for training CNN
image_side = 128
channels = 3
learning_rate = 0.0001
epochs = 20


def generate_data(data_path):
    """
    Generate training and testing data from image directory
    :param data_path: Directory containing 2 folders, with_mask and without_mask
    :return: Training data, training labels, validation data, validation labels
    """

    # Lists to hold images and corresponding labels
    image_data = []
    labels = []

    # Get list of all image names in directory with_mask
    path = data_path + "\with_mask"
    images_in_dir = os.listdir(path)

    # For each image
    for i in images_in_dir:

        # Read image from path, convert to grayscale, and resize it to 224, 224
        image = load_img(path + "\\" + i, target_size=(image_side, image_side))
        image = img_to_array(image)
        image = preprocess_input(image)

        # Add image and corresponding label
        image_data.append(image)
        labels.append(0)

    # Get list of all image names in directory without_mask
    path = data_path + "\without_mask"
    images_in_dir = os.listdir(path)

    # For each image
    for i in images_in_dir:

        # Read image from path, convert to grayscale, and resize it to 224, 224
        image = load_img(path + "\\" + i, target_size=(image_side, image_side))
        image = img_to_array(image)
        image = preprocess_input(image)

        # Add image and corresponding label
        image_data.append(image)
        labels.append(1)

    # Normalize data and shape it correctly
    image_data = np.array(image_data, np.float32)
    labels = np.array(labels)

    encoder = LabelBinarizer()

    labels = encoder.fit_transform(labels)
    labels = to_categorical(labels)

    # Split data to training and validation sets
    x_train, x_test, y_train, y_test = train_test_split(image_data, labels, test_size=0.2, random_state=5)

    return x_train, x_test, y_train, y_test


def train_model(x_train, x_test, y_train, y_test):

    aug = ImageDataGenerator(
        rotation_range=20,
        zoom_range=0.15,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.15,
        horizontal_flip=True,
        fill_mode="nearest")


    base_model = MobileNetV2(weights="imagenet", include_top=False,
                            input_tensor=Input(shape=(image_side, image_side, channels)))

    head_model = base_model.output
    head_model = AveragePooling2D(pool_size=(3, 3))(head_model)
    head_model = Flatten(name="flatten")(head_model)
    head_model = Dense(128, activation="relu")(head_model)
    head_model = Dropout(0.5)(head_model)
    head_model = Dense(2, activation="softmax")(head_model)


    model = Model(inputs=base_model.input, outputs=head_model)

    for layer in base_model.layers:
        layer.trainable = False

    optimizer = Adam(lr=learning_rate, decay=learning_rate / epochs)

    model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])

    history = model.fit(
        aug.flow(x_train, y_train, batch_size=64),
        steps_per_epoch=len(x_train) // 64,
        validation_data=(x_test, y_test),
        validation_steps=len(x_test) // 64,
        epochs=epochs
    )

    model.save(filepath="model_MNet_test", save_format="h5", overwrite=True)

    # Plot training and validation accuracy
    plt.figure(0)
    plt.plot(history.history["accuracy"], color="b")
    plt.plot(history.history["val_accuracy"], color="g")
    plt.title("Model Accuracy")
    plt.legend(["Train", "Validation"])

    # Plot training and validation loss
    plt.figure(1)
    plt.plot(history.history["loss"], color="b")
    plt.plot(history.history["val_loss"], color="g")
    plt.title("Model Loss")
    plt.legend(["Train", "Validation"])

    plt.show()

    return model


def usage():
    """
    Print correct usage of program
    :return:
    """

    print("""
    Syntax:\n
    python Train_MobileNet_V2 <data_path>
    """)

def main():

    if len(sys.argv) < 2:
        usage()
        sys.exit(-1)


    # Generate training and validation data, and create model
    x_train, x_test, y_train, y_test = generate_data(sys.argv[1])
    train_model(x_train, x_test, y_train, y_test)

if __name__ == "__main__":
    main()