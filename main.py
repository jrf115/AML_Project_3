"""
Applied Machine Learning Project_3: Using Neural Networks
John Fahringer
This program tries to build a feedforward neural network (multilayer perceptron)
based on the Wisconsin breast cancer dataset.
"""
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.datasets import load_breast_cancer
load = load_breast_cancer()



wisconsin = load_breast_cancer()
X = wisconsin.data
y = wisconsin.target
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)


#print(wisconsin.shape)
print("Wisconsin data: \n" , X)
print("Wisconsin target: \n" , y)

# Single Hidden Layer Network
from sklearn.neural_network import MLPClassifier
units = 1
nnclf = MLPClassifier(hidden_layer_sizes=[units], solver='lbfgs', random_state=1).fit(X_train, y_train)
print('Accuracy of NN classifier on training set: {:.2f}'.
    format(nnclf.score(X_train, y_train)))
print('Accuracy of NN classifier on test set: {:.2f}'.
    format(nnclf.score(X_test, y_test)))
predict = nnclf.predict(X_test)
print(confusion_matrix(y_test, predict))
