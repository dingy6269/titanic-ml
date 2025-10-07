import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

# print(train.head(3))

X = train.drop(columns=["Survived"])
y = train['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)

print(type(X_train))

X_train_num = X_train.select_dtypes(include=['int64', 'float64'])
X_test_num = X_test.select_dtypes(include=['int64', 'float64'])

scaler = StandardScaler()
scaler.fit(X_train_num)

X_train_scaled = scaler.transform(X_train_num)
X_test_scaled = scaler.transform(X_test_num)

# print(X)
# print(Y)

print("mean_: ", scaler.mean_)
print("scale_: ", scaler.scale_)

print('До маштаб: ', X_train_num.mean(axis=0))
print('После маштаб: ', X_train_scaled.mean(axis=0))

print('------')

print(X_train_num.mean(axis=0))
print(X_train_scaled.mean(axis=0))