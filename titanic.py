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

# print(train_df.columns.values)

# dtype object there
# print(train_df.columns.values)

# print(train_df.head())


if DO_PLOTS:
    import seaborn as sns, matplotlib.pyplot as plt

    sns.set_theme()
    
    # First grid
    g = sns.FacetGrid(train_df, col='Survived')
    # distribution between Age and amount
    g.map(plt.hist, 'Age', bins = 20)
    plt.show()
    
    # Second grid
    grid = sns.FacetGrid(train_df, col='Survived', row='Pclass', aspect=1.6)
    grid.map(plt.hist, 'Age', alpha=.5, bins=20)    
    plt.show()
    
    # Third grid 
    grid = sns.FacetGrid(train_df, row='Embarked', aspect=1.6)
    grid.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', palette='deep')
    grid.add_legend()
    plt.show()

    # Fourth grid
    grid = sns.FacetGrid(train_df, row='Embarked', col='Survived', aspect=1.6)
    grid.map(sns.barplot, 'Sex', 'Fare', alpha=0.5)
    grid.add_legend()
    plt.show()


guess_ages = np.zeros((2, 3))

for df in combine: 
    title_mapping = {"Mr" : 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5} 
    title_replacement = ["Lady","Countess","Capt","Col","Don","Dr","Major","Rev","Sir","Jonkheer","Dona"]

    # name to only prefix like Mr.
    df["Title"] = df["Name"].str.extract(r" ([A-Za-z]+)\.", expand=False)
    
    # replacing all the "rare" types
    df["Title"] = df["Title"].replace(title_replacement, "Rare")
    df["Title"] = df["Title"].replace({"Mlle":"Miss","Ms":"Miss","Mme":"Mrs"}) 
    
    # mapping stuff
    df["Title"] = df["Title"].map(title_mapping)
    df["Title"] = df["Title"].fillna(0).astype(int)
    df["Sex"] = df["Sex"].map({ 'female': 1, 'male': 0 }).astype(int)
    
    for i in range(0, 2):
        # including 0, not including 3
        for j in range(0, 3):
            # there sex and pclass are matche
            # pick age and drop nans
            # it is ROWS not ROW
            # btw got KeyError here
            guess_df = df[(df['Sex'] == i) & (df['Pclass'] == j + 1)]['Age'].dropna()
            
            age_mean = guess_df.mean()
            age_std = guess_df.std()
            age_guess = rnd.uniform(age_mean - age_std, age_mean + age_std)
            
            # print(f"mean: {age_mean}")
            # print(f"std: {age_std}")
            # print(f"age_guess: {age_guess}")
            
            age_guess = guess_df.median()
            
            # print(f"age_guess: {age_guess}")
            
            # print(guess_ages)
            
            # To round the imputed age to the nearest 0.5. (reduce wierd numbers)
            guess_ages[i, j] = int( age_guess / 0.5 + 0.5) * 0.5
    
    for i in range(0, 2):
        for j in range(0, 3):
            df.loc[df.Age.isnull() & (df.Sex == i) & df.Pclass == j + 1, \
                'Age'] = guess_ages[i, j]
            
    
    df['Age'] = df['Age'].fillna(0).astype(int)
            
            
print(train_df.head())

# 2 rows, 3 columns



# print(train_df[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False))
