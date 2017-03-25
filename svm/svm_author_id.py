#!/usr/bin/python

"""
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:
    Sara has label 0
    Chris has label 1
"""

import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess

### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()

### train classifier
from sklearn.svm import SVC
clf = SVC(kernel='linear')
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
