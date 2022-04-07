import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from matplotlib.colors import ListedColormap

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

X_set, y_set = xTrain, yTrain
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() +1, step = 0.01),
        np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() +1, step = 0.01))
plt.contourf (X1, X2, classfier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
    alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
        c = ListedColormap(('red', 'green'))(i), label = j)
plt.title("Naive Bayes (Training set)")
plt.xlabel( "Age")
plt.ylabel( "Estimated Salary")
plt.legend()
plt.show()