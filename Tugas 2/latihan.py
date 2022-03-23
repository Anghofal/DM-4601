import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

dataset = pd.read_csv("latihan.csv")
# iloc[baris,kolom]
# 1 berarti hanya di index 1 
# 1:3 berarti index 1 sampai 2 
# : berarti semua
# :2 berarti index 0 sampai 1
# :-2 berarti index 0 sampai index -2 (0,1)
# :-1 berarti index 0 sampai index -1 (0,1,2)
# 1: berarti index 1 sampai index terakhir
x = dataset.iloc[:,0:3].values
y = dataset.iloc[:,-1].values
#mengisi missing value
imputer = SimpleImputer(missing_values=np.nan,strategy="mean")
imputer.fit(x[:,1:3])
x[:,1:3] = imputer.transform(x[:,1:3])
#me encode x
ct = ColumnTransformer(transformers=[("encoder",OneHotEncoder(),[0])],remainder="passthrough")
x = np.array(ct.fit_transform(x))
#me encode y
le = LabelEncoder()
y = le.fit_transform(y)
#mendapatkan training set dan test set
xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size=0.2,random_state=1)
#men scale data
sc = StandardScaler()

xTrain[:, 3:] = sc.fit_transform(xTrain[:,3:])
xTest[:, 3:] = sc.transform(xTest[:,3:])

print(xTrain)

