import numpy as np
import pandas as pd
from sklearn import tree

irisdata = pd.read_csv("data.csv",delimiter =",",header=0)
irisdata["species"] = pd.factorize(irisdata.Species)[0]

irisdata = irisdata.drop(labels="Id",axis = 1)

irisdata = irisdata.to_numpy()

model = tree.DecisionTreeClassifier()

model = model.fit(inputTraining, labelTraining)

hasilPrediksi = model.predict(inputTesting)

dataTraining = np.concatenate((irisdata[0:30,:],
                                irisdata[50:90,:]),axis=0)

inputTraining = dataTraining[:,0:4]
labelTraining =dataTraining[:,4]

print("label =",labelTraining)
print("hasilPrediksi =", hasilPrediksi)

prediksiBenar = (hasilPrediksi == labelTraining).sum()
prediksiSalah = (hasilPrediksi != labelTraining).sum()

print("akurasi =" ,prediksiBenar/(prediksiBenar+prediksiSalah)*100,"%")