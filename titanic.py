# data analysis and wrangling
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


DO_PLOTS = False


train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

combine = [train_df, test_df]

if DO_PLOTS:
    import seaborn as sns, matplotlib.pyplot as plt

    sns.set_theme()
    
    g = sns.FacetGrid(train_df, col='Survived')
    g.map(plt.hist, 'Age', bins = 20)
    plt.show()
    
    grid = sns.FacetGrid(train_df, col='Survived', row='Pclass', aspect=1.6)
    grid.map(plt.hist, 'Age', alpha=.5, bins=20)    
    plt.show()
    
    grid = sns.FacetGrid(train_df, row='Embarked', aspect=1.6)
    grid.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', palette='deep')
    grid.add_legend()
    plt.show()

    grid = sns.FacetGrid(train_df, row='Embarked', col='Survived', aspect=1.6)
    grid.map(sns.barplot, 'Sex', 'Fare', alpha=0.5)
    grid.add_legend()
    plt.show()


for df in combine: 
    title_mapping = {"Mr" : 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5} 
    title_replacement = ["Lady","Countess","Capt","Col","Don","Dr","Major","Rev","Sir","Jonkheer","Dona"]

    df["Title"] = df["Name"].str.extract(r" ([A-Za-z]+)\.", expand=False)
    
    df["Title"] = df["Title"].replace(title_replacement, "Rare")
    df["Title"] = df["Title"].replace({"Mlle":"Miss","Ms":"Miss","Mme":"Mrs"}) 
    
    df["Title"] = df["Title"].map(title_mapping).fillna(0).astype(int)
    df["Sex"] = df["Sex"].map({ 'female': 1, 'male': 0 }).astype(int)
    
    
    guess_ages = np.zeros((2, 3))
    
    for i in range(0, 2):
        for j in range(0, 3):
            guess_df = df[(df['Sex'] == i) & (df['Pclass'] == j + 1)]['Age'].dropna()
            age_guess = guess_df.median()
            guess_ages[i, j] = int( age_guess / 0.5 + 0.5) * 0.5
    
    for i in range(0, 2):
        for j in range(0, 3):
            df.loc[df.Age.isnull() & (df.Sex == i) & df.Pclass == j + 1, \
                'Age'] = guess_ages[i, j]
            
    
    df['Age'] = df['Age'].fillna(0).astype(int)
            
            

# more features
for df in combine:
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    df["IsAlone"] = (df["FamilySize"] == 1).astype(int)
    df["Age*Class"] = df["Age"] * df["Pclass"]


for df in combine:
    embarked_mapping = {"S": 0, "C": 1, "Q": 2}
    
    df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])
    df["Embarked"] = df["Embarked"].map(embarked_mapping).astype(int)


test_df["Fare"] = test_df["Fare"].fillna(test_df["Fare"].median())
q1, q2, q3 = train_df["Fare"].quantile([0.25, 0.50, .75]).tolist()

for df in combine:
    df.loc[df["Fare"] <= q1, "Fare"] = 0
    df.loc[(df["Fare"] > q1) & (df["Fare"] <= q2), "Fare"] = 1
    df.loc[(df["Fare"] > q2) & (df["Fare"] <= q3), "Fare"] = 2
    df.loc[df["Fare"] > q3, "Fare"] = 3
    df["Fare"] = df["Fare"].astype(int)

for df in combine:
    df.drop(columns=["Ticket", "Cabin", "Name"], inplace=True)

X = train_df.drop(columns=["Survived", "PassengerId"])
y = train_df["Survived"]
XTest = test_df.drop(columns=["PassengerId"])


# from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

# model = SVC(kernel="sigmoid")
model = RandomForestClassifier(
    n_estimators=500,
    max_depth=6,
    random_state=42
)

model.fit(X, y)
prediction = model.predict(XTest)

submission = pd.DataFrame({
    "PassengerId": test_df["PassengerId"],
    "Survived": prediction
})

submission.to_csv("submission.csv", index=False)

print("Wrote submission.csv")