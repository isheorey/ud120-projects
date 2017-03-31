#!/usr/bin/python

"""
    Implementation of AdaBoost algorithm on terrain dataset

    Optional passed arguments:
    * n_estimators (default=50)
"""
import sys
import matplotlib.pyplot as plt
from time import time
from prep_terrain_data import makeTerrainData
from class_vis import prettyPicture

### process arguments
ada_n_estimators = 50
if len(sys.argv) > 1:
    n_estimators = int(sys.argv[1])

print "# of estimators:", ada_n_estimators

### generate terrain data
features_train, labels_train, features_test, labels_test = makeTerrainData()

### the training data (features_train, labels_train) have both "fast" and "slow"
### points mixed together--separate them so we can give them different colors
### in the scatterplot and identify them visually
grade_fast = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==0]
bumpy_fast = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==0]
grade_slow = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==1]
bumpy_slow = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==1]

### initial visualization
plt.xlim(0.0, 1.0)
plt.ylim(0.0, 1.0)
plt.scatter(bumpy_fast, grade_fast, color = "b", label="fast")
plt.scatter(grade_slow, bumpy_slow, color = "r", label="slow")
plt.legend()
plt.xlabel("bumpiness")
plt.ylabel("grade")
plt.show()

### train classifier
from sklearn.ensemble import AdaBoostClassifier
clf = AdaBoostClassifier(n_estimators=ada_n_estimators)
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

### plot pretty picture
try:
    prettyPicture(clf, features_test, labels_test)
except NameError:
    pass
