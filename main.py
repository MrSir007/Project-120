import csv
import pandas as pd
from sklearn import datasets
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split as tts
from sklearn.metrics import accuracy_score as acc
from sklearn.preprocessing import StandardScaler

wine = datasets.load_wine()
head = wine.head()
print(head)

X = wine[["Min","Max"]]
Y = wine["Hue"]
xTrain, xTest, yTrain, yTest = tts(X, Y, test_size=0.25, random_state=36)
ss = StandardScaler()
xTrain = ss.fit_transform(xTrain)
xTest = ss.fit_transform(xTest)
gnb = GaussianNB()
gnb.fit(xTrain, yTrain)
yPrediction = gnb.predict(xTest)
accuracy = acc(yTest, yPrediction)
print(accuracy)