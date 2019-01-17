# USAGE
# python train.py --dataset dataset

# set the matplotlib backend so figures can be saved in the background
import matplotlib

matplotlib.use("Agg")

import argparse
from datetime import datetime
import json
import os
import pickle
import random

import cv2
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.optimizers import Adam
from keras.preprocessing.image import img_to_array, ImageDataGenerator
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
from pyimagesearch.smallervggnet import SmallerVGGNet
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split



# initialize the number of epochs to train for, initial learning rate,
# batch size, and image dimensions
EPOCHS = 75
INIT_LR = 1e-3
BATCH_SIZE = 32
IMAGE_DIMS = (96, 96, 3)
RANDOM_SEED = 42
TIME_STRING_FORMAT = "%Y%m%d-%H%M"

def load_image_and_label_files(random_seed):
    # grab the image paths and randomly shuffle them
    print("[INFO] loading images...")
    # read the labels file
    imgdir = "../images/candidate_images"
    labelf = "../labelbox/cjol5nvoox1wy07127npy8ajt%2Fcjol5oih8x2f90712z5gzhnh3%2Fexport-2018-11-18T17_50_06.412Z.json"
    with open(labelf, "r") as f:
        labelbox = json.load(f)

    label_map = {}
    for l in labelbox:
        name = l["External ID"]
        path = os.path.join(imgdir, name)
        labels = l["Label"]
        if labels == "Skip" or labels.get("questionable_(reason?)"):
            continue
        print(labels)

        props = [
            # NOTE: We don't have a lot of data yet...
            #
            # My understanding is that currently the model treats each combination of labels as completely independent.
            # For example, a "1 empty red diamond" is treated as having no relation to "1 empty red squiggle".
            # In reality, the images share all but one property, so they aren't really indepedent.
            #
            # A problem with this approach is that we don't yet have many examples per case, once we get so fine grained.
            # Let the number of labels be `l`, the number of samples be `N`, and recall that there are three sub-types per label
            # That means we have:
            #
            #     samples per label combination = (N/(3**m))
            #
            # If we reduce the number of labels, we'll have a more examples per combination.
            # Ideally, we can just get enough labed data, but that may require 100s of examples per combination.
            #
            # Let's see how the performance looks and continue to iterate!
            "color",
            "shape",
            "number",
            "fill",
        ]
        label_list = []
        for p in props:
            label_list.append("{}_{}".format(p, labels[p]))
        label_map[name] = label_list

    image_paths = sorted(list(paths.list_images(imgdir)))
    random.seed(random_seed)
    random.shuffle(image_paths)

    return image_paths, label_map


def preprocess_data(image_paths, label_map):
    # initialize the data and labels
    data = []
    labels = []
    
    # loop over the input images
    for image_path in image_paths:
        # load the image, pre-process it, and store it in the data list
        image = cv2.imread(image_path)
        image = cv2.resize(image, (IMAGE_DIMS[1], IMAGE_DIMS[0]))
        image = img_to_array(image)
    
        # extract set of class labels from the image path and update the
        # labels list
        _, filename = os.path.split(image_path)
        l = label = label_map.get(filename)
        if label:
            data.append(image)
            labels.append(l)
    
    # scale the raw pixel intensities to the range [0, 1]
    data = np.array(data, dtype="float") / 255.0
    print(
        "[INFO] data matrix: {} images ({:.2f}MB)".format(
            len(image_paths), data.nbytes / (1024 * 1000.0)
        )
    )
    
    # binarize the labels using scikit-learn's special multi-label
    # binarizer implementation
    print("[INFO] class labels:")
    mlb = MultiLabelBinarizer()
    labels = mlb.fit_transform(labels)

    # loop over each of the possible class labels and show them
    for (i, label) in enumerate(mlb.classes_):
        print("{}. {}".format(i + 1, label))
    return labels, data, mlb


def run_model(train_x, train_y, test_x, test_y, num_output_classes, model_output_name='cards_model_output'):

    # val_loss is cross-validation loss, loss is testing loss
    checkpoint_callback = ModelCheckpoint(model_output_name + ".model", monitor='val_loss', save_best_only=True)
    # stops the fit after {patience} epochs where the the variable {monitor} doesn't change my more than {min_delta}
    stopping_callback = EarlyStopping(monitor='val_loss', min_delta=0, patience=10)
    callbacks_list = [checkpoint_callback, stopping_callback]

    # construct the image generator for data augmentation
    aug = ImageDataGenerator(
        rotation_range=25,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode="nearest",
    )

    # initialize the optimizer (SGD is sufficient)
    opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)

    # initialize the model using a sigmoid activation as the final layer
    # in the network so we can perform multi-label classification
    print("[INFO] compiling model...")
    model = SmallerVGGNet.build(
        width=IMAGE_DIMS[1],
        height=IMAGE_DIMS[0],
        depth=IMAGE_DIMS[2],
        classes=num_output_classes,
        finalAct="sigmoid",
    )

    # compile the model using binary cross-entropy rather than
    # categorical cross-entropy -- this may seem counterintuitive for
    # multi-label classification, but keep in mind that the goal here
    # is to treat each output label as an independent Bernoulli
    # distribution
    model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])

    # train the network
    print("[INFO] training network...")
    fitted_model = model.fit_generator(
        aug.flow(train_x, train_y, batch_size=BATCH_SIZE),
        validation_data=(test_x, test_y),
        steps_per_epoch=len(train_x) // BATCH_SIZE,
        epochs=EPOCHS,
        verbose=1,
        callbacks=callbacks_list
    )
    return fitted_model


def plot_model_summary(plot_filename, fitted_model):
    # plot the training loss and accuracy
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(np.arange(0, len(fitted_model.history["loss"])), fitted_model.history["loss"], label="train_loss")
    plt.plot(np.arange(0, len(fitted_model.history["val_loss"])), fitted_model.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, len(fitted_model.history["acc"])), fitted_model.history["acc"], label="train_acc")
    plt.plot(np.arange(0, len(fitted_model.history["val_acc"])), fitted_model.history["val_acc"], label="val_acc")
    plt.title("Training Loss and Accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="upper left")
    plt.savefig(plot_filename)


if __name__ == "__main__":
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "-d",
        "--dataset",
        required=True,
        help="path to input dataset (i.e., directory of images)",
    )
    
    args = vars(ap.parse_args())
    random_seed = 42
    for i in range(3):
        random_seed += i
        run_name = "%s_model_run_seed_%s" % (datetime.utcnow().strftime(TIME_STRING_FORMAT), random_seed)

        image_paths, label_map = load_image_and_label_files(random_seed)
        labels, data, label_binarizer = preprocess_data(image_paths, label_map)

        # partition the data into training and testing splits using 80% of
        # the data for training and the remaining 20% for testing
        train_x, test_x, train_y, test_y = train_test_split(data,
                                                            labels,
                                                            test_size=0.2,
                                                            random_state=random_seed)

        fitted_model = run_model(train_x,
                                 train_y,
                                 test_x,
                                 test_y,
                                 num_output_classes=len(label_binarizer.classes_),
                                 model_output_name=os.path.join("models", run_name)
                                 )

        plot_model_summary(os.path.join("models", run_name + ".plot.png"), fitted_model)
        with open(os.path.join("models", run_name + '_data'), 'wb') as fout:
            pickle.dump((train_x, test_x, train_y, test_y), fout)

        # plot classification auc
        # plot classification error by data type

        # save the multi-label binarizer to disk
        print("[INFO] serializing label binarizer...")
        with open(os.path.join("models", run_name + '_labels'), "wb") as fout:
            pickle.dump(label_binarizer, fout)
