from __future__ import print_function

import numpy as np
import h5py
import json
import matplotlib.pyplot as plt
import warnings

from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.model_selection import KFold

warnings.simplefilter(action="ignore", category=FutureWarning)

# load the user configs
with open('1_deep/conf/conf.json') as f:
    config = json.load(f)

# config variables
test_size = config["test_size"]
seed = config["seed"]
features_path = config["features_path"]
labels_path = config["labels_path"]
results = config["results"]
classifier_path = config["classifier_path"]
train_path = config["train_path"]
num_classes = config["num_classes"]

num_trees = 100
scoring = "accuracy"

models = []
results_list = []
names = []

# import features and labels
h5f_data = h5py.File(features_path, 'r')
h5f_label = h5py.File(labels_path, 'r')
features_string = h5f_data['dataset_1']
labels_string = h5f_label['dataset_1']
features = np.array(features_string)
labels = np.array(labels_string)
h5f_data.close()
h5f_label.close()

# create classifiers
models.append(('LR', LogisticRegression(random_state=seed)))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier(random_state=seed)))
models.append(('RF', RandomForestClassifier(
    n_estimators=num_trees, random_state=seed)))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(random_state=seed, probability=True)))

# verify the shape of features and labels
print("[INFO] features shape: {}".format(features.shape))
print("[INFO] labels shape: {}".format(labels.shape))
print("[INFO] training started...")

# split the training and testing data
trainData, testData, trainLabels, testLabels = train_test_split(
    np.array(features), np.array(labels), test_size=test_size, random_state=seed)

print("[INFO] splitted train and test data...")
print("[INFO] train data  : {}".format(trainData.shape))
print("[INFO] test data   : {}".format(testData.shape))
print("[INFO] train labels: {}".format(trainLabels.shape))
print("[INFO] test labels : {}".format(testLabels.shape))

f = open(results, "w")
for name, model in models:
    print("[INFO] creating classifier: {}".format(name))
    # perform 10-fold cross validation
    kfold = KFold(n_splits=10)
    cv_results = cross_val_score(
        model, trainData, trainLabels, cv=kfold, scoring=scoring)
    results_list.append(cv_results)
    names.append(name)
    msg = "%f (%f)" % (cv_results.mean(), cv_results.std())
    f.write("Model: {}\n".format(str(name)))
    f.write("10-fold cross validation accuracy 'mean + standard deviation': {}\n".format(str(msg)))

    model.fit(trainData, trainLabels)
    rank_1 = 0  # rank 1 accuracy
    rank_5 = 0  # rank 5 accuracy

    for (label, features) in zip(testLabels, testData):     # loop over test data
        # predict the probability of each class label and take the top-5 class labels
        predictions = model.predict_proba(np.atleast_2d(features))[0]
        predictions = np.argsort(predictions)[::-1][:5]

        # rank-1 prediction increment
        if label == predictions[0]:
            rank_1 += 1

        # rank-5 prediction increment
        if label in predictions:
            rank_5 += 1

    # convert accuracies to percentages
    rank_1 = (rank_1 / float(len(testLabels))) * 100
    rank_5 = (rank_5 / float(len(testLabels))) * 100

    # write the accuracies to file
    f.write("Rank-1 accuracy: {:.2f}%\n".format(rank_1))
    f.write("Rank-5 accuracy: {:.2f}%\n\n".format(rank_5))

    # evaluate the model of test data
    preds = model.predict(testData)

    # write the classification report to file
    # f.write("{}\n".format(classification_report(testLabels, preds, labels=np.unique(preds), digits=4)))
    f.write("{}\n".format(classification_report(testLabels, preds, digits=4)))
    f.write("\n\n")
f.close()

# boxplot algorithm comparison
fig = plt.figure()
fig.suptitle('Machine Learning algorithm comparison')
ax = fig.add_subplot(111)
plt.boxplot(results_list)
ax.set_xticklabels(names)
plt.show()
