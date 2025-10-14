import pandas as pd
import numpy as np
import random as rnd

# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier

import matplotlib.pyplot as plt


DO_PLOTS = False

train = pd.read_csv("game_of_thrones_train.csv")
test = pd.read_csv("game_of_thrones_test.csv")

#
# logic
# 

if DO_PLOTS:
    train['popularity'].hist(bins=50, figsize=(15, 20))

    plt.xlabel('Popularity')
    plt.ylabel('Count')
    plt.title('Distribution of Popularity')
    plt.grid(False)

    plt.show()


train['popularity'] = (train['popularity'].fillna(0).gt(0)).astype('int8')


# print(train['popularity'])

train['age_value'] = (train['age'].fillna(0))
train['age_no_data'] = [1 if np.isnan(x) else 0 for x in train['age']]


print(train.value_counts())

# train['culture'].str.lower()



train.groupby('male', dropna = False)['isAlive'].mean()


median = train.groupby('male', dropna = False)['isAlive'].median()


num_cols = train.select_dtypes(include=['int64','float64']).dropna(axis=1).columns

num_data = train.select_dtypes(include=['int64', 'float64'])
# x axis
num_cols = num_data.dropna(axis=1).columns


corr_matrix = train[num_cols].corr()

print(corr_matrix)