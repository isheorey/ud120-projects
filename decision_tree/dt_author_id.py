#!/usr/bin/python

"""
    This is the code to accompany the Lesson 3 (decision tree) mini-project.

    Use a Decision Tree to identify emails from the Enron corpus by author:
    Sara has label 0
    Chris has label 1

    Optional passed arguments:
    * sampling_percent (default=100)
    * min_samples_split (default=2)
"""

import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess

### process arguments
sampling_pct = 100
dt_min_samples_split = 2
if len(sys.argv) > 1:
    sampling_pct = int(sys.argv[1])
if len(sys.argv) > 2:
    dt_min_samples_split = int(sys.argv[2])

print "sampling %:", sampling_pct
print "min samples split:", dt_min_samples_split

### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()

### sample dataset
features_train_end = len(features_train) * sampling_pct / 100
labels_train_end = len(labels_train) * sampling_pct / 100
features_train = features_train[:features_train_end]
labels_train = labels_train[:labels_train_end]
print "# of features:", len(features_train[0])

### train classifier
from sklearn import tree
clf = tree.DecisionTreeClassifier(min_samples_split=dt_min_samples_split)
t0_train = time()   # start training timer
clf.fit(features_train, labels_train)
print "training time:", round(time() - t0_train, 3), "s"
t0_pred = time()    # start prediction timer
pred = clf.predict(features_test)
print "prediction time:", round(time() - t0_pred, 3), "s"

# determine accuracy
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(labels_test, pred)
print "accuracy:", accuracy
