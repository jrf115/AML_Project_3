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
topTen = []
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
for layers in [[5], [10], [5,10], [10,5], [10,20], [5,10,20], [5,20,10]]:
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

                    count += 1
                    print("**************************", str(count), "**************************")

                    one_hidden_nnclf = MLPClassifier(hidden_layer_sizes=layers, solver=gradient_solve,
                                                     activation=activation_func, alpha=regularization_param,
                                                     random_state=1).fit(X_train_scale_unscale, y_train)
                    trainAcc = one_hidden_nnclf.score(X_train_scale_unscale, y_train)
                    print('Accuracy of',  len(layers), 'hidden layers NN classifier on training set: {:.2f}'.
                        format(trainAcc))
                    print('Accuracy of',  len(layers), 'hidden layers NN classifier on test set: {:.2f}'.
                        format(one_hidden_nnclf.score(X_test_scale_unscale, y_test)))
                    one_hidden_predict = one_hidden_nnclf.predict(X_test_scale_unscale)
                    print(confusion_matrix(y_test, one_hidden_predict))
                    # print("String:", str(layers), "\n")
                    modelDict = {
                        "Model": "hiddenLayers = " + str(layers) + " gradientSolver = " + gradient_solve +
                                 " activationFunction = " + activation_func + " reguParam = " + str(regularization_param) +
                                 " Scale/Unscale = " + scale_unscale,
                        "Train_Acc": trainAcc,
                        "Test_Acc": (one_hidden_nnclf.score(X_test_scale_unscale, y_test)),
                        "CM": str(confusion_matrix(y_test, one_hidden_predict))
                    }
                    if count == 1:
                        topTen.append(modelDict)
                    elif count <= 10:
                        listCount = 0
                        for t in topTen:
                            if t.get("Train_Acc") < trainAcc:
                                topTen.insert(listCount, modelDict)
                                break
                            elif t.get("Train_Acc") == trainAcc:
                                topTen.append(modelDict)
                                break
                            listCount += 1
                    else:
                        listCount = 0
                        for t in topTen:
                            if t.get("Train_Acc") < trainAcc:
                                topTen.pop()
                                topTen.insert(listCount, modelDict)
                                break
                            listCount += 1


print("\n*****\nComputed", count, "models.\n*****")
print("\n\nTop Ten Models: \n")
num = 1
for t in topTen:
    print(str(num) + ") " + t.get("Model"))
    print("with a training Acc: {:.2f}" . format(t.get("Train_Acc")))
    print("and a Testing Acc: {:.2f}" . format(t.get("Test_Acc")))
    print("and a Confusion Matrix of:\n", t.get("CM"))
    print()
    num += 1