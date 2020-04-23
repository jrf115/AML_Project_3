"""
Applied Machine Learning Project_3: Using Neural Networks
John Fahringer
This program tries to build a feedforward neural network (multilayer perceptron)
based on the Wisconsin breast cancer dataset.
"""
import pandas as pd
from sklearn.datasets import load_breast_cancer
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import RegionPlot as rp

wisconsin = pd.read_csv('data/wdbc.data', ',')
X = wisconsin[[
'Mean_Radius','Mean_Texture','Mean_Perimeter','Mean_Area','Mean_Smoothness','Mean_Compactness','Mean_Concavity','Mean_Concave_Points','Mean_Symmetry','Mean_Fractal_Dimension',
'Radius','Texture','Perimeter','Area','Smoothness','Compactness','Concavity','Concave_Points','Symmetry','Fractal_Dimension',
'Worst_Radius','Worst_Texture','Worst_Perimeter','Worst_Area','Worst_Smoothness','Worst_Compactness','Worst_Concavity','Worst_Concave_Points','Worst_Symmetry','Worst_Fractal_Dimension'
]]
y = wisconsin['Diagnosis']
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)


print(wisconsin.shape)
print("Wisconsin data: \n" , X)
print("Wisconsin target: \n" , y)