import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

diabetes = pd.read_csv('Pima_Indian_diabetes.csv')
diabetes["ModifiedPregnancies"] = diabetes["Pregnancies"]
diabetes["ModifiedPregnancies"].fillna(-1, inplace=True)
plt.figure(figsize=(15, 8))
sns.distplot(diabetes["ModifiedPregnancies"], bins = 30)
plt.show()
# print (diabetes.head(25))

count = diabetes["Pregnancies"].isnull().sum()
    diabetes["Pregnancies"] = diabetes["Pregnancies"].abs().round()
    diabetes["ModifiedPregnancies"] = diabetes["Pregnancies"]
    diabetes["ModifiedPregnancies"].fillna(0, inplace=True)
    sumOfPregnancies = diabetes["ModifiedPregnancies"].sum()
    TotalNonNull = len(diabetes["Pregnancies"]) - count
    Average = (int(float(sumOfPregnancies)/TotalNonNull))
    diabetes["FinalPregnancies"] = diabetes["Pregnancies"]
    diabetes["FinalPregnancies"].fillna(Average, inplace=True)
    x = [x for x in range(len(diabetes["Pregnancies"]))]
    plt.plot(x, diabetes["FinalPregnancies"])
    plt.show()