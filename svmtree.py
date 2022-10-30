#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Author: Gael Varoquaux <gael dot varoquaux at normalesup dot org>
# License: BSD 3 clause

# Standard scientific Python imports
from math import gamma
import matplotlib.pyplot as plt

# Import datasets, classifiers and performance metrics
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
import numpy as np
import pandas as pd

train_frac = [0.5, 0.6, 0.7, 0.8, 0.9]
test_frac = [0.25, 0.2, 0.15, 0.1, 0.05]
dev_frac = [0.25, 0.2, 0.15, 0.1, 0.05]

GAMMA = 0.001
c = 10.0
#train_frac = 0.8
#test_frac = 0.1
#dev_frac = 0.1

# In[7]:


digits = datasets.load_digits()

_, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
for ax, image, label in zip(axes, digits.images, digits.target):
    ax.set_axis_off()
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
    ax.set_title("Training: %i" % label)


# In[ ]:


# flatten the images
n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))

svm_acc_train = []
svm_acc_test = []
svm_acc_dev = []

tree_acc_train = []
tree_acc_test = []
tree_acc_dev = []

print("Confusion Matrix to compare lables of SVM and Decision Tree")
for i in range(len(train_frac)):
    
        dev_test_frac = 1-train_frac[i]
        X_train, X_dev_test, y_train, y_dev_test = train_test_split(
            data, digits.target, test_size=dev_test_frac, shuffle=True)

# Split data into 50% train and 50% test subsets
        X_test, X_dev, y_test, y_dev = train_test_split(
            X_dev_test,y_dev_test, test_size=(dev_frac[i])/dev_test_frac, shuffle=True)

        # Create a classifier: a support vector classifier
        clf = svm.SVC()

        clf_model = DecisionTreeClassifier(criterion="gini", random_state=42,max_depth=6, min_samples_leaf=1)  

        clf.set_params(gamma = 0.001)

        # Learn the digits on the train subset
        clf.fit(X_train, y_train)

        svm_predicted_train = clf.predict(X_dev_test)
        svm_predicted_dev = clf.predict(X_dev)
        svm_predicted_test = clf.predict(X_test)

        clf_model.fit(X_train, y_train)

        tree_predicted_train = clf_model.predict(X_dev_test)
        tree_predicted_dev = clf_model.predict(X_dev)
        tree_predicted_test = clf_model.predict(X_test)

        print("Confusion Matrix test:"+str(i))
        print(confusion_matrix(tree_predicted_train[0:80], svm_predicted_train[0:80], labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]))
        print("Confusion Matrix dev:"+str(i))
        print(confusion_matrix(tree_predicted_dev[0:80], svm_predicted_train[0:80], labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]))
        print("Confusion Matrix train:"+str(i))
        print(confusion_matrix(tree_predicted_test[0:80], svm_predicted_train[0:80], labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]))

        svm_accu = metrics.accuracy_score(y_pred = svm_predicted_dev, y_true = y_dev)
        svm_accu_test = metrics.accuracy_score(y_pred = svm_predicted_test, y_true = y_test)
        svm_accu_train = metrics.accuracy_score(y_pred = svm_predicted_train, y_true = y_dev_test)
        svm_acc_train.append(svm_accu_train)
        svm_acc_test.append(svm_accu_test)
        svm_acc_dev.append(svm_accu)

        tree_accu = metrics.accuracy_score(y_pred = tree_predicted_dev, y_true = y_dev)
        tree_accu_test = metrics.accuracy_score(y_pred = tree_predicted_test, y_true = y_test)
        tree_accu_train = metrics.accuracy_score(y_pred = tree_predicted_train, y_true = y_dev_test)
        tree_acc_train.append(tree_accu_train)
        tree_acc_test.append(tree_accu_test)
        tree_acc_dev.append(tree_accu)
        

        # Predict the value of the digit on the test subset
svm_predicted = clf.predict(X_test)
tree_predicted = clf_model.predict(X_test)


_, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
for ax, image, prediction in zip(axes, X_test, svm_predicted):
    ax.set_axis_off()
    image = image.reshape(8, 8)
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
    ax.set_title(f"Prediction: {prediction}")


print(
    f"Classification report for classifier {clf}:\n"
    f"{metrics.classification_report(y_test, svm_predicted)}\n"
)

_, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
for ax, image, prediction in zip(axes, X_test, tree_predicted):
    ax.set_axis_off()
    image = image.reshape(8, 8)
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
    ax.set_title(f"Prediction: {prediction}")


print(
    f"Classification report for classifier {clf}:\n"
    f"{metrics.classification_report(y_test, tree_predicted)}\n"
)

svm_acc_trains = []
svm_acc_tests = []
svm_acc_devs = []
tree_acc_trains = []
tree_acc_tests = []
stree_acc_devs = []
svm_acc_train.append(np.mean(svm_acc_train))
svm_acc_train.append(np.std(svm_acc_train))

svm_acc_test.append(np.mean(svm_acc_test))
svm_acc_test.append(np.std(svm_acc_test))

svm_acc_dev.append(np.mean(svm_acc_dev))
svm_acc_dev.append(np.std(svm_acc_dev))

tree_acc_train.append(np.mean(tree_acc_train))
tree_acc_train.append(np.std(tree_acc_train))

tree_acc_test.append(np.mean(tree_acc_test))
tree_acc_test.append(np.std(tree_acc_test))

tree_acc_dev.append(np.mean(tree_acc_dev))
tree_acc_dev.append(np.std(tree_acc_dev))

#print(svm_acc_train)
#print(svm_acc_test)
#print(svm_acc_dev)
#print(tree_acc_train)
#print(tree_acc_test)
#print(tree_acc_dev)

df = pd.DataFrame({'SVM Train':svm_acc_train,'SVM Test':svm_acc_test,'SVM Dev':svm_acc_dev,
'Tree Train':tree_acc_train,'Tree Test':tree_acc_test,'Tree Dev':tree_acc_dev},index = ['1', '2', '3', '4', '5', 'Mean', 'STD'])

print(df)