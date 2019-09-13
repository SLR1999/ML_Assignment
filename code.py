import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import math
from sklearn.preprocessing import normalize
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA 
from outlier import MahalanobisDist, MD_detectOutliers, is_pos_def

diabetes = pd.read_csv('Pima_Indian_diabetes.csv')
columnNames = ["Glucose", "Insulin", "SkinThickness", "BMI", "DiabetesPedigreeFunction", "BloodPressure", "Pregnancies", "Age"]
columnNamesIntegersPositive = ["Pregnancies", "Age"]
len_1 = 0
columnNamesFloatPositive = ["Glucose", "Insulin", "SkinThickness", "BMI", "DiabetesPedigreeFunction", "BloodPressure"]
len_2 = 0
for i in range(0, len(columnNames)):
    colName = columnNames[i]
    if (colName == "Glucose" or colName == "BMI" or colName == "BloodPressure"):
        diabetes[colName] = diabetes[colName].replace(0, np.NaN)
    count = diabetes[colName].isnull().sum()
    if (colName in columnNamesIntegersPositive):
        diabetes[colName] = diabetes[colName].abs().round()
    else : 
        diabetes[colName] = diabetes[colName].abs()
    diabetes["Modified" + colName] = diabetes[colName]
    diabetes["Modified" + colName].fillna(0, inplace=True)
    sumOfValues = diabetes["Modified" + colName].sum()
    TotalNonNull = len(diabetes[colName]) - count
    Average = (int(float(sumOfValues)/TotalNonNull))
    diabetes["Final" + colName] = diabetes[colName]
    diabetes["Final" + colName].fillna(Average, inplace=True)
    # x = [x for x in range(len(diabetes[colName]))]
    # plt.title("Modified"+ colName)
    # plt.plot(x, diabetes["Modified"+ colName])
    # plt.show()
    # plt.title("Final"+ colName)
    # plt.plot(x, diabetes["Final" + colName])
    # plt.show()

data = np.array([diabetes["FinalGlucose"], diabetes["FinalInsulin"], diabetes["FinalSkinThickness"], diabetes["FinalBMI"], diabetes["FinalDiabetesPedigreeFunction"], diabetes["FinalBloodPressure"], diabetes["FinalPregnancies"], diabetes["FinalAge"]])
data = data.T
outcome = np.array(diabetes["Outcome"])

X_train, X_test, y_train, y_test = train_test_split(data, outcome, stratify=outcome, random_state=random.randint(13,5231))

outliers_indices = MD_detectOutliers(X_train)

print("Outliers Indices: {}\n".format(outliers_indices))
print("Outliers:")
print(len(outliers_indices))

newData = []
newOutcome = []
for var in range(len(X_train)):
    if var not in outliers_indices:
        newData.append(X_train[var])
        newOutcome.append(y_train[var])
print (len(newOutcome))
newData = np.array(newData)
newOutcome = np.array(newOutcome)
newData = normalize(newData,return_norm=True)[0]
print(newData.shape)
logreg = LogisticRegression().fit(X_train, y_train)
print("Training set score: {:.3f}".format(logreg.score(X_train, y_train)))
print("Test set score: {:.3f}".format(logreg.score(X_test, y_test)))

