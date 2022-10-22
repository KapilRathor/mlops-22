#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Author: Gael Varoquaux <gael dot varoquaux at normalesup dot org>
# License: BSD 3 clause

# Standard scientific Python imports
import matplotlib.pyplot as plt

# Import datasets, classifiers and performance metrics
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split

gamma_list = [0.01, 0.005, 0.001, 0.0005, 0.0001]
c_list = [0.1, 0.2, 0.5, 0.7, 1, 2, 5, 7, 10]

GAMMA = 0.001
c = 1.0
train_frac = 0.8
test_frac = 0.1
dev_frac = 0.1


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

dev_test_frac = 1-train_frac
X_train, X_dev_test, y_train, y_dev_test = train_test_split(
    data, digits.target, test_size=dev_test_frac, shuffle=True
)



# Split data into 50% train and 50% test subsets
X_test, X_dev, y_test, y_dev = train_test_split(
    X_dev_test,y_dev_test, test_size=(dev_frac)/dev_test_frac, shuffle=True
)

h_param_comb = [{'gamma':g, 'C':c} for g in gamma_list for c in c_list]

best_acc = -1.0
best_mod = None
best_h_param = None
for h_params in h_param_comb:

        # Create a classifier: a support vector classifier
        clf = svm.SVC()

        hyper_params = h_params
        clf.set_params(**hyper_params)

        # Learn the digits on the train subset
        clf.fit(X_train, y_train)

        predicted_train = clf.predict(X_dev_test)
        predicted_dev = clf.predict(X_dev)
        predicted_test = clf.predict(X_test)

        accu = metrics.accuracy_score(y_pred = predicted_dev, y_true = y_dev)
        accu_test = metrics.accuracy_score(y_pred = predicted_test, y_true = y_test)
        accu_train = metrics.accuracy_score(y_pred = predicted_train, y_true = y_dev_test)

        if accu > best_acc:
            best_acc = accu
            best_mod = clf
            best_h_param  = h_params
            print("New best params:"+str(h_params))
            print("New best train accuracy:"+str(accu_train))
            print("New best dev accuracy:"+str(accu))
            print("New best test accuracy:"+str(accu_test))


        # Predict the value of the digit on the test subset
predicted = best_mod.predict(X_test)


_, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
for ax, image, prediction in zip(axes, X_test, predicted):
    ax.set_axis_off()
    image = image.reshape(8, 8)
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
    ax.set_title(f"Prediction: {prediction}")


print(
    f"Classification report for classifier {clf}:\n"
    f"{metrics.classification_report(y_test, predicted)}\n"
)

print("Best hyperparameters are:")
print(h_params)

