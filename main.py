# Part 1: Python Configuration and Data Loading

import sys
if sys.version_info[0] < 3:
    raise Exception("Python 3 not detected.")

import numpy as np
import matplotlib.pyplot as plt
import csv
from sklearn import svm
from scipy import io

for data_name in ["mnist", "spam", "cifar10"]:
    data = io.loadmat("data/%s_data.mat" % data_name)
    print("\nloaded %s data!" % data_name)
    fields = "test_data", "training_data", "training_labels"
    for field in fields:
        print(field, data[field].shape)


# Part 2: Data Partitioning

np.random.seed(69)

# mnist
mnist_data = io.loadmat("data/%s_data.mat" % "mnist")
mnist_train_data = mnist_data["training_data"] / 255
mnist_train_labels = mnist_data["training_labels"]

# shuffle both mnist training data and training labels
indices = np.arange(mnist_train_data.shape[0])
np.random.shuffle(indices)
mnist_train_data = mnist_train_data[indices]
mnist_train_labels = mnist_train_labels[indices]

# divide mnist training data into validation and training sets
mnist_val_set_sz = 59900 # size of mnist validation set
mnist_val_set = mnist_train_data[0:mnist_val_set_sz] # mnist validation set
mnist_val_set_labels = mnist_train_labels[0:mnist_val_set_sz] # mnist validation set labels
mnist_train_set = mnist_train_data[mnist_val_set_sz:] # mnist training set
mnist_train_set_labels = mnist_train_labels[mnist_val_set_sz:] # mnist training set labels

# verify mnist validation set and training set sizes
print("mnist validation set has size", mnist_val_set.shape)
print("mnist training set has size", mnist_train_set.shape)

# plot one mnist sample point
# mnist_height = 28
# mnist_width = 28
# A = mnist_train_data[1].reshape(mnist_height,mnist_width)
# plt.matshow(A)
# plt.show()

# spam
spam_data = io.loadmat("data/%s_data.mat" % "spam")
spam_train_data = spam_data["training_data"]
spam_train_labels = spam_data["training_labels"]

# shuffle both spam training data and training labels
indices = np.arange(spam_train_data.shape[0])
np.random.shuffle(indices)
spam_train_data = spam_train_data[indices]
spam_train_labels = spam_train_labels[indices]

# divide spam training data into validation and training sets
# spam_val_set_sz = spam_train_data.shape[0]//5 # size of spam validation set
spam_val_set_sz = spam_train_data.shape[0] - 100 # size of spam validation set
spam_val_set = spam_train_data[0:spam_val_set_sz] # spam validation set
spam_val_set_labels = spam_train_labels[0:spam_val_set_sz] # spam validation set labels
spam_train_set = spam_train_data[spam_val_set_sz:] # spam training set
spam_train_set_labels = spam_train_labels[spam_val_set_sz:] # spam training set labels

# verify spam validation set and training set sizes
print("spam validation set has size", spam_val_set.shape)
print("spam training set has size", spam_train_set.shape)

# print one spam sample point
# print(spam_train_data[4])

# cifar10
cifar10_data = io.loadmat("data/%s_data.mat" % "cifar10")
cifar10_train_data = cifar10_data["training_data"] / 255
cifar10_train_labels = cifar10_data["training_labels"]

# shuffle both cifar10 training data and training labels
indices = np.arange(cifar10_train_data.shape[0])
np.random.shuffle(indices)
cifar10_train_data = cifar10_train_data[indices]
cifar10_train_labels = cifar10_train_labels[indices]

# divide cifar10 training data into validation and training sets
cifar10_val_set_sz = 49900 # size of cifar10 validation set
cifar10_val_set = cifar10_train_data[0:cifar10_val_set_sz] # cifar10 validation set
cifar10_val_set_labels = cifar10_train_labels[0:cifar10_val_set_sz] # cifar10 validation set labels
cifar10_train_set = cifar10_train_data[cifar10_val_set_sz:] # cifar10 training set
cifar10_train_set_labels = cifar10_train_labels[cifar10_val_set_sz:] # cifar10 training set labels

# verify cifar10 validation set and training set sizes
print("cifar10 validation set has size", cifar10_val_set.shape)
print("cifar10 training set has size", cifar10_train_set.shape)

# plot one cifar10 sample point
# cifar10_height = 32;
# cifar10_width = 32;
# cifar10_area = cifar10_height * cifar10_width
# A = cifar10_train_data[12]
# red = np.empty((cifar10_height,cifar10_width))
# green = np.empty((cifar10_height,cifar10_width))
# blue = np.empty((cifar10_height,cifar10_width))
# rgb = np.empty((cifar10_height,cifar10_width,3))
# for i in range(cifar10_height):
#     for j in range(cifar10_width):
#         red[i,j] = A[i * cifar10_height + j]
#         green[i,j] = A[cifar10_area + i * cifar10_height + j]
#         blue[i,j] = A[2* cifar10_area + i * cifar10_height + j]
#         rgb[i,j] = [red[i,j], green[i,j], blue[i,j]]
# plt.imshow(rgb)
# plt.show()

# Part 3: Support Vector Machines: Coding

# SVM on mnist
# clf = svm.SVC(kernel='linear')
# print(mnist_train_set.shape)
# print(mnist_train_set_labels.shape)
# clf.fit(mnist_train_set, mnist_train_set_labels.ravel())
# mnist_train_results = clf.predict(mnist_train_set)
# mnist_val_results = clf.predict(mnist_val_set)
# mnist_train_accuracy = (mnist_train_set_labels.ravel() == mnist_train_results).sum()/mnist_train_set_labels.shape[0]
# mnist_val_accuracy = (mnist_val_set_labels.ravel() == mnist_val_results).sum()/mnist_val_set_labels.shape[0]
# print("mnist classifier has accuracy", mnist_train_accuracy, "on the training set data")
# print("mnist classifier has accuracy", mnist_val_accuracy, "on the validation set data")

# x=[]
# y=[]
# with open('q3_mnist_accuracies_vs_num_training_samples.csv','r') as csvfile:
#     plots = csv.reader(csvfile)
#     for row in plots:
#         x.append(int(row[0]))
#         y.append(int(row[1]))
#
# plt.plot(x,y, label='Loaded from file!')
# plt.xlabel('x')
# plt.ylabel('y')

# SVM on spam
# clf = svm.SVC(kernel='linear')
# print(spam_train_set.shape)
# print(spam_train_set_labels.shape)
# clf.fit(spam_train_set, spam_train_set_labels.ravel())
# spam_train_results = clf.predict(spam_train_set)
# spam_val_results = clf.predict(spam_val_set)
# spam_train_accuracy = (spam_train_set_labels.ravel() == spam_train_results).sum()/spam_train_set_labels.shape[0]
# spam_val_accuracy = (spam_val_set_labels.ravel() == spam_val_results).sum()/spam_val_set_labels.shape[0]
# print("spam classifier has accuracy", spam_train_accuracy, "on the training set data")
# print("spam classifier has accuracy", spam_val_accuracy, "on the validation set data")

# SVM on cifar10
# clf = svm.SVC(kernel='linear')
# print(cifar10_train_set.shape)
# print(cifar10_train_set_labels.shape)
# clf.fit(cifar10_train_set, cifar10_train_set_labels.ravel())
# cifar10_train_results = clf.predict(cifar10_train_set)
# cifar10_val_results = clf.predict(cifar10_val_set)
# cifar10_train_accuracy = (cifar10_train_set_labels.ravel() == cifar10_train_results).sum()/cifar10_train_set_labels.shape[0]
# cifar10_val_accuracy = (cifar10_val_set_labels.ravel() == cifar10_val_results).sum()/cifar10_val_set_labels.shape[0]
# print("cifar10 classifier has accuracy", cifar10_train_accuracy, "on the training set data")
# print("cifar10 classifier has accuracy", cifar10_val_accuracy, "on the validation set data")

# Part 4: Hyperparameter Tuning

# Part 5: K-Fold Cross-Validation

# Part 6: Kaggle

