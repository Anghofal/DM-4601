import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
#trainning dan test set
dataset = pd.read_csv("dataaa.csv")
x = dataset.iloc[:,[2,3]].values
y = dataset.iloc[:,-1]

xTrain , xTest , yTrain , yTest = train_test_split(x,y,test_size=0.25,random_state=0)
# scaling
sc = StandardScaler()
xTrain = sc.fit_transform(xTrain)
xTest = sc.fit_transform(xTest)
#
classfier = GaussianNB()
classfier.fit(xTrain,yTrain)
#
yPred = classfier.predict(xTest)
#
cm = confusion_matrix(yTest,yPred)
print(cm)
