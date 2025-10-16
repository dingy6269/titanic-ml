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

import seaborn as sns, matplotlib.pyplot as plt

DO_PLOTS = False
LOGS = False

train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")


# doing describing stuff

# print(train_df.info())
# print(train_df.describe())

# print(train_df.head())

# observing visual coorelations between SalePrice and other fields


if DO_PLOTS:
    print(sns.distplot(train_df['SalePrice']))
    print(sns.barplot(x='YearBuilt', y='SalePrice', data=train_df))

    sns.barplot(x='SaleCondition', y='SalePrice', data=train_df)

    sns.barplot(x='YrSold', y='SalePrice', data=train_df)
    
    plt.show()


# dropping all the BS (non related to table)

train_df=train_df.drop("Id",axis=1)
train_df=train_df.drop("Alley",axis=1)
train_df=train_df.drop("PoolQC",axis=1)
train_df=train_df.drop("Fence",axis=1)
train_df=train_df.drop("MiscFeature",axis=1)

# from test also (makes sense)
test_df=test_df.drop("Alley",axis=1)
test_df=test_df.drop("PoolQC",axis=1)
test_df=test_df.drop("Fence",axis=1)
test_df=test_df.drop("MiscFeature",axis=1)


# not spot NA values in the train.csv
# found them?
# NOW we need to fill NA values

train_df["LotFrontage"] = train_df["LotFrontage"].fillna(train_df["LotFrontage"].mean())
train_df["MasVnrArea"] = train_df["MasVnrArea"].fillna(train_df["MasVnrArea"].mean())
train_df["GarageYrBlt"] = train_df["GarageYrBlt"].fillna(train_df["GarageYrBlt"].mean())


if LOGS:
    print(train_df.select_dtypes(include=[np.number]).columns)

### Train Categorical 
# In pandas, categorical columns are usually stored as object dtype — which means strings.
# so replace NA values

c = ("GarageType", "GarageFinish", "GarageQual", "GarageCond", "BsmtFinType2", "BsmtCond", 
     "BsmtQual", "BsmtExposure", "MasVnrType", "Electrical", "FireplaceQu", "BsmtFinType1")

for col in c:
    if train_df[col].dtype == "object":
        train_df[col] = train_df[col].fillna("None")
        
        
# same stuff, replace numerical on test csv
        
test_df["LotFrontage"] = test_df["LotFrontage"].fillna(test_df["LotFrontage"].mean())
test_df["MasVnrArea"] = test_df["MasVnrArea"].fillna(test_df["MasVnrArea"].mean())
test_df["GarageYrBlt"] = test_df["GarageYrBlt"].fillna(2001)
test_df["GarageCars"] = test_df["GarageCars"].fillna(0)
test_df["GarageArea"] = test_df["GarageArea"].fillna(test_df["GarageArea"].mean())
test_df["BsmtFullBath"] = test_df["BsmtFullBath"].fillna(0)
test_df["BsmtHalfBath"] = test_df["BsmtHalfBath"].fillna(0)
test_df["BsmtFinSF1"] = test_df["BsmtFinSF1"].fillna(test_df["BsmtFinSF1"].mean())
test_df["BsmtFinSF2"] = test_df["BsmtFinSF2"].fillna(test_df["BsmtFinSF2"].mean())
test_df["TotalBsmtSF"] = test_df["TotalBsmtSF"].fillna(test_df["TotalBsmtSF"].mean())
test_df["BsmtUnfSF"] = test_df["BsmtUnfSF"].fillna(test_df["BsmtUnfSF"].mean())


# same with catecorial. replace NA

c = ("GarageType", "GarageFinish", "GarageQual", "GarageCond", "BsmtFinType2", "BsmtCond", "BsmtQual", "BsmtExposure", "MasVnrType", "Electrical","MSZoning","Utilities","Exterior1st","Exterior2nd","KitchenQual","Functional","FireplaceQu","SaleType", "BsmtFinType1")
for col in c:
  if test_df[col].dtype == "object":
    test_df[col] = test_df[col].fillna("None")
    
    
if LOGS:
    print(test_df.info())
    
    
### LabelEncoder


print(train_df.head())

from sklearn.preprocessing import LabelEncoder

catagory_cols = ('MSZoning','Street','LotShape','LandContour','Utilities','LotConfig','LandSlope','Neighborhood','Condition1','Condition2','BldgType', 'HouseStyle', 'RoofStyle','RoofMatl','Exterior1st','Exterior2nd','ExterCond','Foundation','Heating','HeatingQC','CentralAir','KitchenQual','Functional','FireplaceQu','PavedDrive','SaleType','SaleCondition', "GarageType", "GarageFinish", "GarageQual", "GarageCond", "BsmtFinType2", "BsmtCond", "BsmtQual", "BsmtExposure", "MasVnrType", "Electrical", "BsmtFinType1", "ExterQual")


for c in catagory_cols:
    le = LabelEncoder()
    train_df[c] = le.fit_transform(train_df[c].values)

# Because machine-learning models can only handle numeric input.
# thats why

# Spliting the Train & Test datasets¶


catagory_cols = ('MSZoning','Street','LotShape','LandContour','Utilities','LotConfig','LandSlope','Neighborhood','Condition1','Condition2','BldgType', 'HouseStyle', 'RoofStyle','RoofMatl','Exterior1st','Exterior2nd','ExterCond','Foundation','Heating','HeatingQC','CentralAir','KitchenQual','Functional','FireplaceQu','PavedDrive','SaleType','SaleCondition', "GarageType", "GarageFinish", "GarageQual", "GarageCond", "BsmtFinType2", "BsmtCond", "BsmtQual", "BsmtExposure", "MasVnrType", "Electrical", "BsmtFinType1", "ExterQual")


for c in catagory_cols:
    le = LabelEncoder()
    test_df[c] = le.fit_transform(test_df[c].values)
    
if LOGS:
    print(train_df.head())

# Id: pure identifier. Carries no causal signal about price. 
# # Including it lets models pick up fake numeric patterns and overfit. It also won’t generalize.

# supervised learning, f(x) => y

# ENDGAME

# inputs
# contains all columns except SalePrice.
X_train = train_df.drop("SalePrice", axis=1)
# answers
Y_train = train_df["SalePrice"]

# axis =1 => columns
X_test = test_df.drop("Id", axis=1).copy()
X_test = X_test.reindex(columns=X_train.columns)
assert list(X_test.columns) == list(X_train.columns)

# FEATURE SCALING


# Make features comparable. A $100000 feature won’t dominate a 0–10 feature.

from sklearn.preprocessing import StandardScaler

# Scale features so no column dominates, optimization is stable, and distance/regularization make sense across all features.
sc = StandardScaler()

# fit_transform on X_train learns the scaling params  and applies them.
# transform on X_test applies those same params to test so there’s no data leakage 
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


if LOGS:
    print(X_train)
    

# PCA — это способ упростить данные.
# Сдвигаем все признаки так, чтобы их среднее было 0.
# Находим направления, где данные “раскиданы” сильнее всего.
# Берём первые k таких направлений.
# Проецируем данные на них. Получаем меньше признаков, но почти ту же информацию.

# A component in PCA = one direction in feature space that captures as much variance as possible.
from sklearn.decomposition import PCA

pca = PCA(n_components=10)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)

# core end logic

from sklearn.ensemble import RandomForestRegressor

regressor = RandomForestRegressor(n_estimators=200, random_state=0)

regressor.fit(X_train, Y_train)
Y_pred = regressor.predict(X_test)

from sklearn.metrics import accuracy_score
# regressor.score(X_train, Y_train)
# 97.97
# regressor = round(regressor.score(X_train, Y_train) * 100, 2)


submission = pd.DataFrame({
    "Id": test_df["Id"],
    "SalePrice": Y_pred
})

submission.to_csv("submission.csv", index=False)