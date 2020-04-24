"""
Applied Machine Learning Project_3: Using Neural Networks
John Fahringer
This program tries to build a feedforward neural network (multilayer perceptron)
based on the Wisconsin breast cancer dataset.
"""
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

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

fig, subaxes = plt.subplots(3, 1, figsize=(6, 18))
for units, axis in zip([1, 10, 100], subaxes):
    nnclf = MLPClassifier(hidden_layer_sizes=[units], solver='lbfgs', random_state=1).fit(X_train, y_train)
    title = "Wisconsin Dataset: Neural Net Classifier, 1 layer, {} units".format(units)
    rp.plot_class_regions_for_classifier_subplot(nnclf, X_train, y_train, X_test, y_test, title, axis)
    plt.tight_layout()