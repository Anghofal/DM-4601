import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import *
from sklearn.model_selection import train_test_split


#memasukkan data
dataset = pd.read_csv("data.csv")

#
x = dataset.iloc[:,:-1].values

#
y = dataset.iloc[:,[2]].values

#mengisi missing value
imputer = SimpleImputer(missing_values=np.nan,strategy="mean")
imputer.fit(y)
y = imputer.transform(y)
#print(y)

#me encode x(encode string ke numberik)
ct = ColumnTransformer(transformers=[("encoder",OneHotEncoder(),[0,1])],remainder="passthrough")
x = np.array(ct.fit_transform(x))
#print(x)

#training set dan test set
xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size=0.2,random_state=1)
#print(xTrain)
#print(yTest)

#men scale data
sc = StandardScaler()
xTrain[:, :] = sc.fit_transform(xTrain[:,:])
xTest[:, :] = sc.transform(xTest[:,:])
yTrain[:, :] = sc.fit_transform(yTrain[:,:])
yTest[:, :] = sc.transform(yTest[:,:])
print(yTrain)