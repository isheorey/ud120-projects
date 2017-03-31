#!/usr/bin/python

"""
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:
    Sara has label 0
    Chris has label 1

    Optional passed arguments:
    * sampling_percent (default=100)
    * kernel (default='linear')
    * C (default=1.0)
"""

import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess

### process arguments
sampling_pct = 100
svm_kernel = 'linear'
svm_C = 1.0
if len(sys.argv) > 1:
    sampling_pct = int(sys.argv[1])
if len(sys.argv) > 2:
    svm_kernel = sys.argv[2]
if len(sys.argv) > 3:
    svm_C = float(sys.argv[3])

print "sampling %:", sampling_pct
print "kernel:", svm_kernel
print "C:", svm_C

### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()

### sample dataset
features_train_end = len(features_train) * sampling_pct / 100
labels_train_end = len(labels_train) * sampling_pct / 100
features_train = features_train[:features_train_end]
labels_train = labels_train[:labels_train_end]

### train classifier
from sklearn.svm import SVC
clf = SVC(C=svm_C, kernel=svm_kernel)
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

# predicted results
print "prediction for element 10:", pred[10]
print "prediction for element 26:", pred[26]
print "prediction for element 50:", pred[50]
print "# of events predicted to be Chris class:", pred.sum()
