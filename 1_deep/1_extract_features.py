''' https://gogul.dev/software/flower-recognition-deep-learning
1. Prepare the training dataset with heatmap images and its corresponding labels.
2. Specify your own configurations in conf.json file.
3. Extract and store features from the last fully connected layers (before 'predictions' layer which contains last 'softmax' classifier) (or intermediate layers) of a pre-trained Deep Neural Net (CNN) using extract_features.py.
4. Train a Machine Learning model such as Logisitic Regression (and other like SVM, NB, etc.) using these CNN extracted features and labels using train.py.
5. Evaluate the trained model on unseen data and make further optimizations if necessary.
'''

import time
import datetime
import json
import os
import h5py
import glob
import numpy as np
import warnings
import tensorflow as tf

from sklearn.preprocessing import LabelEncoder
from keras.layers import Input
from keras import layers
from keras.models import Model, Sequential
from keras.utils.image_utils import load_img, img_to_array
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.applications.mobilenet import MobileNet, preprocess_input
from keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input
from keras.applications.resnet import ResNet50, preprocess_input
from keras.applications.xception import Xception, preprocess_input
from keras.applications.vgg19 import VGG19, preprocess_input
from keras.applications.vgg16 import VGG16, preprocess_input

warnings.simplefilter(action="ignore", category=FutureWarning)

# load the user configs
with open('1_deep/conf/conf.json') as f:
    config = json.load(f)

# config variables
model_name = config["model"]
train_path = config["train_path"]
features_path = config["features_path"]
labels_path = config["labels_path"]
test_size = config["test_size"]
results = config["results"]
model_path = config["model_path"]
weights = None  # None - train from the scratch; 'imagenet' - use pre-trained model
include_top = True  # False - take features from any intermediate layer; True - take features before fully connected layers

# variables to hold features and labels
features = []
labels = []

# start time
print("[STATUS] start time - {}".format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M")))
start = time.time()

# create the pretrained models; check for pretrained weight usage or not; check for top layers to be included or not
if model_name == "vgg16":   # nowe
    base_model = VGG16(weights=weights)
    model = Model(inputs=base_model.input,
                  outputs=base_model.get_layer('fc2').output)
    image_size = (224, 224)
elif model_name == "vgg19": # nowe
    base_model = VGG19(weights=weights)
    model = Model(inputs=base_model.input,
                  outputs=base_model.get_layer('fc2').output)
    image_size = (224, 224)
elif model_name == "resnet50":  # nowe
    base_model = ResNet50(weights=weights)
    model = Model(inputs=base_model.input,
                  outputs=base_model.get_layer('avg_pool').output)
    image_size = (224, 224)
elif model_name == "inceptionv3":   #nowe
    base_model = InceptionV3(
        include_top=include_top, weights=weights, input_tensor=Input(shape=(299, 299, 3)))
    model = Model(inputs=base_model.input,
                  outputs=base_model.get_layer('avg_pool').output)
    image_size = (299, 299)
elif model_name == "inceptionresnetv2": # nowe
    base_model = InceptionResNetV2(
        include_top=include_top, weights=weights, input_tensor=Input(shape=(299, 299, 3)))
    model = Model(inputs=base_model.input,
                  outputs=base_model.get_layer('avg_pool').output)
    image_size = (299, 299)
elif model_name == "mobilenet": # nowe
    base_model = MobileNet(include_top=include_top, weights=weights, input_tensor=Input(
        shape=(224, 224, 3)), input_shape=(224, 224, 3))
    model = Model(inputs=base_model.input,
                  outputs=base_model.get_layer('reshape_2').output)
    image_size = (224, 224)
elif model_name == "xception":  # nowe
    base_model = Xception(weights=weights)
    model = Model(inputs=base_model.input,
                  outputs=base_model.get_layer('avg_pool').output)
    image_size = (299, 299)
elif model_name == "sequential":
    model = Sequential([
        layers.Rescaling(1./255, input_shape=(150, 1200, 3)),
        layers.Conv2D(16, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(4)
    ])
    image_size = (150, 1200)
else:
    base_model = None

print("[INFO] successfully loaded base model and model...")
# compile model with optimizer, loss function
#model.compile(optimizer=tf.keras.optimizers.Adam(lr=3e-4),
#                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
#                metrics=['accuracy'])
# model.summary()

# encode the labels
print("[INFO] encoding labels...")
train_labels = os.listdir(train_path)
le = LabelEncoder()
le.fit([tl for tl in train_labels])

# loop over all the labels in the folder and extract features one by one
count = 1
for i, label in enumerate(train_labels):
    cur_path = train_path + "/" + label
    count = 1
    for image_path in glob.glob(cur_path + "/*.jpg"):
        img = load_img(image_path, target_size=image_size)
        x = img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        feature = model.predict(x)
        flat = feature.flatten()
        features.append(flat)
        labels.append(label)
        print("[INFO] processed - " + str(count))
        count += 1
    print("[INFO] completed label - " + label)

# encode the labels using LabelEncoder
le = LabelEncoder()
le_labels = le.fit_transform(labels)

# get the shape of training labels
print("[STATUS] training labels: {}".format(le_labels))
print("[STATUS] training labels shape: {}".format(le_labels.shape))

# save features and labels
h5f_data = h5py.File(features_path, 'w')
h5f_data.create_dataset('dataset_1', data=np.array(features))
h5f_label = h5py.File(labels_path, 'w')
h5f_label.create_dataset('dataset_1', data=np.array(le_labels))
h5f_data.close()
h5f_label.close()

# save model and weights
model_json = model.to_json()
with open(model_path + str(test_size) + ".json", "w") as json_file:
    json_file.write(model_json)

# save weights
model.save_weights(model_path + str(test_size) + ".h5")
print("[STATUS] saved model and weights to disk..")
print("[STATUS] features and labels saved..")

# end time
end = time.time()
print("[STATUS] end time - {}".format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M")))
