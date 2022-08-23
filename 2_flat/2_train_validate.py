import h5py
import numpy as np
import os
import warnings
import matplotlib.pyplot as plt
import json

from sklearn.model_selection import KFold, train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score

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

image_size = tuple((1200, 150))

# -----------------------------------
# TRAINING MODEL
# -----------------------------------

# get the training labels
train_labels = os.listdir(train_path)

# sort the training labels
train_labels.sort()

if not os.path.exists(test_path):
    os.makedirs(test_path)

# create all the machine learning models
models = []
models.append(('LR', LogisticRegression(random_state=seed)))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier(random_state=seed)))
models.append(('RF', RandomForestClassifier(
    n_estimators=num_trees, random_state=seed)))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(random_state=seed, probability=True)))

# variables to hold the results and names
results = []
names = []

# import the feature vector and trained labels
h5f_data = h5py.File(h5_data, 'r')
h5f_label = h5py.File(h5_labels, 'r')

global_features_string = h5f_data['dataset_1']
global_labels_string = h5f_label['dataset_1']

global_features = np.array(global_features_string)
global_labels = np.array(global_labels_string)

h5f_data.close()
h5f_label.close()

# verify the shape of the feature vector and labels
print("[STATUS] features shape: {}".format(global_features.shape))
print("[STATUS] labels shape: {}".format(global_labels.shape))

print("[STATUS] training started...")

# split the training and testing data
trainDataGlobal, testDataGlobal, trainLabelsGlobal, testLabelsGlobal = train_test_split(
    np.array(global_features), np.array(global_labels), test_size=test_size, random_state=seed)

print("[STATUS] splitted train and test data...")
print("Train data  : {}".format(trainDataGlobal.shape))
print("Test data   : {}".format(testDataGlobal.shape))
print("Train labels: {}".format(trainLabelsGlobal.shape))
print("Test labels : {}".format(testLabelsGlobal.shape))


f = open(results1, "w")
for name, model in models:
    print("[INFO] creating classifier: {}".format(name))
    # 10-fold cross validation
    kfold = KFold(n_splits=10)
    cv_results = cross_val_score(
        model, trainDataGlobal, trainLabelsGlobal, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%f (%f)" % (cv_results.mean(), cv_results.std())
    f.write("Model: {}\n".format(str(name)))
    f.write("10-fold cross validation accuracy 'mean + standard deviation': {}\n".format(str(msg)))

    model.fit(trainDataGlobal, trainLabelsGlobal)
    rank_1 = 0  # rank 1 accuracy
    rank_5 = 0  # rank 5 accuracy
    # loop over test data
    for (label, features) in zip(testLabelsGlobal, testDataGlobal):
        # predict the probability of each class label and
        # take the top-5 class labels
        predictions = model.predict_proba(np.atleast_2d(features))[0]
        predictions = np.argsort(predictions)[::-1][:5]

        # rank-1 prediction increment
        if label == predictions[0]:
            rank_1 += 1

        # rank-5 prediction increment
        if label in predictions:
            rank_5 += 1

    # convert accuracies to percentages
    rank_1 = (rank_1 / float(len(testLabelsGlobal))) * 100
    rank_5 = (rank_5 / float(len(testLabelsGlobal))) * 100

    # write the accuracies to file
    f.write("Rank-1 accuracy: {:.2f}%\n".format(rank_1))
    f.write("Rank-5 accuracy: {:.2f}%\n\n".format(rank_5))

    # evaluate the model of test data
    preds = model.predict(testDataGlobal)

    # write the classification report to file
    # f.write("{}\n".format(classification_report(testLabels, preds, labels=np.unique(preds))))
    f.write("{}\n".format(classification_report(testLabelsGlobal, preds, digits=4)))
    f.write("\n\n")
f.close()

# boxplot algorithm comparison
fig = plt.figure()
fig.suptitle('Machine Learning algorithm comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()
