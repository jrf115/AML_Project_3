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

count = 0
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
for gradient_solve in ["lbfgs", "sgd", "adam"]:
    for activation_func in ["logistic", "tanh", "relu"]:
        for regularization_param in [0.01, 0.1, 1.0]:
            for scale_unscale in ["unscaled", "scaled"]:
                if scale_unscale == "scaled":
                    X_train_scale_unscale = scaler.fit_transform(X_train)
                    X_test_scale_unscale = scaler.transform(X_test)
                elif scale_unscale == "unscaled":
                    X_train_scale_unscale = X_train
                    X_test_scale_unscale = X_test
                else:
                    print("\n*********\n*******ERROR: Scale or unscaled were not chosen*******\n*********\n")

                # Single Hidden Layer Network
                one_hidden_nnclf = MLPClassifier(hidden_layer_sizes=[5], solver=gradient_solve,
                                                 activation=activation_func, alpha=regularization_param,
                                                 random_state=1).fit(X_train_scale_unscale, y_train)
                print('Accuracy of one hidden layer NN classifier on training set: {:.2f}'.
                    format(one_hidden_nnclf.score(X_train_scale_unscale, y_train)))
                print('Accuracy of one hidden layer NN classifier on test set: {:.2f}'.
                    format(one_hidden_nnclf.score(X_test_scale_unscale, y_test)))
                one_hidden_predict = one_hidden_nnclf.predict(X_test_scale_unscale)
                print(confusion_matrix(y_test, one_hidden_predict))


                # Two Hidden Layer Network
                two_hidden_nnclf = MLPClassifier(hidden_layer_sizes=[5, 10], solver=gradient_solve,
                                                 activation=activation_func, alpha=regularization_param,
                                                 random_state=1).fit(X_train_scale_unscale, y_train)
                print('Accuracy of two hidden layers NN classifier on training set: {:.2f}'.
                      format(two_hidden_nnclf.score(X_train_scale_unscale, y_train)))
                print("Accuracy of two hidden layer NN classifier on test set: {:.2f}".
                      format(two_hidden_nnclf.score(X_test_scale_unscale, y_test)))
                two_hidden_predict = two_hidden_nnclf.predict(X_test_scale_unscale)
                print(confusion_matrix(y_test, two_hidden_predict))


                # Three Hiddden Layer Network
                three_hidden_nnclf = MLPClassifier(hidden_layer_sizes=[5, 10, 100], solver=gradient_solve,
                                                   activation=activation_func, alpha=regularization_param,
                                                   random_state=1).fit(X_train_scale_unscale, y_train)
                print('Accuracy of three hidden layers NN classifier on training set: {:.2f}'.
                      format(three_hidden_nnclf.score(X_train_scale_unscale, y_train)))
                print("Accuracy of three hidden layer NN classifier on test set: {:.2f}".
                      format(three_hidden_nnclf.score(X_test_scale_unscale, y_test)))
                three_hidden_predict = three_hidden_nnclf.predict(X_test_scale_unscale)
                print(confusion_matrix(y_test, three_hidden_predict))
                count += 3

print("\n*****\nComputed", count, "models.\n*****")