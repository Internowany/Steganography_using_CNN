# https://gogul.dev/software/image-classification-python
# FEATURE EXTRACTION FROM DATASET
# Input image -> Apply Image Descriptor -> Feature Vector

''' global feature descriptors examples:
* Color - Color Channel Statistics (Mean, Standard Deviation) and Color Histogram
* Shape - Hu Moments, Zernike Moments
* Texture - Haralick Texture, Local Binary Patterns (LBP)
* Others - Histogram of Oriented Gradients (HOG), Threshold Adjancency Statistics (TAS)
'''
''' local feature descriptors examples:
SIFT (Scale Invariant Feature Transform)
SURF (Speeded Up Robust Features)
ORB (Oriented Fast and Rotated BRIEF)
BRIEF (Binary Robust Independed Elementary Features)
'''

import time
import datetime
import json
import os
import h5py
import numpy as np
import warnings
import cv2
import mahotas
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

warnings.simplefilter(action="ignore", category=FutureWarning)

# load the user configs
with open('2_flat/conf/conf.json') as f:
    config = json.load(f)

# tunable-parameters
train_path = config["train_path"]
test_path = config["test_path"]
h5_data = config["features_path"]
h5_labels = config["labels_path"]
bins = config["bins"]

# images_per_class = 558
fixed_size = tuple((1200, 150))

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


# start time
print("[STATUS] start time - {}".format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M")))
start = time.time()

# --------------------
# GLOBAL FEATURE EXTRACTION
# --------------------

# get the training labels
# encode the labels
print("[INFO] encoding labels...")
train_labels = os.listdir(train_path)
le = LabelEncoder()
le.fit([tl for tl in train_labels])

# sort the training labels
# train_labels.sort()
# print(train_labels)

# empty lists to hold feature vectors and labels
global_features = []
labels = []

# loop over all the labels in the folder
count = 1
for training_name in train_labels:
    dir = os.path.join(train_path, training_name)
    current_label = training_name
    count = 1
    for filename in os.listdir(dir):
        file = os.path.join(dir, filename)
        image = cv2.imread(file)
        image = cv2.resize(image, fixed_size)
        fv_hu_moments = fd_hu_moments(image)
        fv_haralick = fd_haralick(image)
        fv_histogram = fd_histogram(image)
        global_feature = np.hstack([fv_histogram, fv_haralick, fv_hu_moments])
        global_features.append(global_feature)
        labels.append(current_label)
        print("[INFO] processed - " + str(count))
        count += 1
    print("[INFO] completed label - " + current_label)

# encode the labels using LabelEncoder
targetNames = np.unique(labels)
le = LabelEncoder()
le_labels = le.fit_transform(labels)

# get the shape of training labels
print("[STATUS] training labels: {}".format(le_labels))
print("[STATUS] training labels shape: {}".format(le_labels.shape))

# get the overall feature vector size
print("[STATUS] feature vector size {}".format(
    np.array(global_features).shape))

# get the overall training label size
#print("[STATUS] training Labels {}".format(np.array(labels).shape))

# scale features in the range (0-1)
scaler = MinMaxScaler(feature_range=(0, 1))
rescaled_features = scaler.fit_transform(global_features)
print("[STATUS] feature vector normalized...")

# save the feature vector using HDF5
h5f_data = h5py.File(h5_data, 'w')
h5f_data.create_dataset('dataset_1', data=np.array(rescaled_features))

h5f_label = h5py.File(h5_labels, 'w')
h5f_label.create_dataset('dataset_1', data=np.array(le_labels))

h5f_data.close()
h5f_label.close()

# end time
end = time.time()
print("[STATUS] end time - {}".format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M")))
