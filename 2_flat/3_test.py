import h5py
import numpy as np
import os
import glob
import cv2
import warnings
import json
import mahotas
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler

warnings.simplefilter(action="ignore", category=FutureWarning)

# load the user configs
with open('2_flat/conf/conf.json') as f:
    config = json.load(f)

# tunable-parameters
train_path = config["train_path"]
test_path = config["test_path"]
h5_data = config["features_path"]
h5_labels = config["labels_path"]
results1 = config["results"]
scoring = config["scoring"]
seed = config["seed"]
test_size = config["test_size"]
num_trees = config["num_trees"]
bins = config["bins"]
fixed_size = tuple((1200, 150))

results = []
names = []

# feature-descriptor-1: Hu Moments


def fd_hu_moments(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    feature = cv2.HuMoments(cv2.moments(image)).flatten()
    return feature

# feature-descriptor-2: Haralick Texture


def fd_haralick(image):
    # convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # compute the haralick texture feature vector
    haralick = mahotas.features.haralick(gray).mean(axis=0)
    # return the result
    return haralick

# feature-descriptor-3: Color Histogram


def fd_histogram(image, mask=None):
    # convert the image to HSV color-space
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # compute the color histogram
    hist = cv2.calcHist([image], [0, 1, 2], None, [
                        bins, bins, bins], [0, 256, 0, 256, 0, 256])
    # normalize the histogram
    cv2.normalize(hist, hist)
    # return the histogram
    return hist.flatten()


# -----------------------------------
# TESTING MODEL
# -----------------------------------
# get the training labels
train_labels = os.listdir(train_path)

# sort the training labels
train_labels.sort()
h5f_data = h5py.File(h5_data, 'r')
h5f_label = h5py.File(h5_labels, 'r')
global_features_string = h5f_data['dataset_1']
global_labels_string = h5f_label['dataset_1']

global_features = np.array(global_features_string)
global_labels = np.array(global_labels_string)
trainDataGlobal, testDataGlobal, trainLabelsGlobal, testLabelsGlobal = train_test_split(
    np.array(global_features), np.array(global_labels), test_size=test_size, random_state=seed)

# create the model - Logistic Regression
clf = LogisticRegression(random_state=seed)

# fit the training data to the model
clf.fit(trainDataGlobal, trainLabelsGlobal)

# loop through the test images
for file in glob.glob(test_path + "/*.jpg"):
    # read the image
    image = cv2.imread(file)

    # resize the image
    image = cv2.resize(image, fixed_size)

    ####################################
    # Global Feature extraction
    ####################################
    fv_hu_moments = fd_hu_moments(image)
    fv_haralick = fd_haralick(image)
    fv_histogram = fd_histogram(image)

    ###################################
    # Concatenate global features
    ###################################
    global_feature = np.hstack([fv_histogram, fv_haralick, fv_hu_moments])
    global_feature = global_feature.reshape(-1, 1)
    global_feature = global_feature.transpose()
    print("[STATUS] feature vector size {}".format(
        np.array(global_feature).shape))

    # scale features in the range (0-1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    rescaled_feature = scaler.fit_transform(global_feature)

    # predict label of test image
    prediction = clf.predict(rescaled_feature)[0]
    print(train_labels)
    print(prediction)

    # show predicted label on image
    cv2.putText(image, train_labels[prediction], (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 3)

    # display the output image
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.show()
