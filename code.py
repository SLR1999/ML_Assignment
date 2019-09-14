import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA 
from outlier import MahalanobisDist, MD_detectOutliers
from confusionMatrix import plot_confusion_matrix

'''Reading csv file into a dataframe named diabetes'''
diabetes = pd.read_csv('Pima_Indian_diabetes.csv')

'''Plotting the heatmap of correlation matrix'''
sns.heatmap(diabetes.corr(),cmap='Blues',annot=False)

'''Features as columns'''
columnNames = ["Glucose", "Insulin", "BMI", "DiabetesPedigreeFunction", "SkinThickness", "BloodPressure", "Pregnancies", "Age"]

'''Features which can take Positive Integral Values'''
columnNamesIntegersPositive = ["Pregnancies", "Age"]

'''Features which can take Positive Float Values'''
columnNamesFloatPositive = ["Glucose", "Insulin", "BMI", "SkinThickness", "DiabetesPedigreeFunction", "BloodPressure"]

'''
Cleaning the data and handling zero values by:
   1. Changing zeros to NaN values in Glucose, BMI and Blood Pressure
   2. Calculating average for non NaN values for all the features
   3. Changing all the NaNs to average value for all the features
'''
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

    '''
    Plotting the values for each data point to analyse how to fill in the missing data
    '''
    # x = [x for x in range(len(diabetes[colName]))]
    # plt.title("Modified"+ colName)
    # plt.plot(x, diabetes["Modified"+ colName])
    # plt.show()
    # plt.title("Final"+ colName)
    # plt.plot(x, diabetes["Final" + colName])
    # plt.show()

'''Making a numpy array (of all the features) after filling in missing data and removing zero outliers'''
data = np.array([diabetes["FinalGlucose"], diabetes["FinalInsulin"], diabetes["FinalBMI"], diabetes["FinalDiabetesPedigreeFunction"], diabetes["FinalSkinThickness"], diabetes["FinalBloodPressure"], diabetes["FinalPregnancies"], diabetes["FinalAge"]])
data = data.T

'''Outcome vector'''
outcome = np.array(diabetes["Outcome"])

'''Normalizing the data for PCA (but not using it to build the model)'''
# data = normalize(data,return_norm=True)[0]

'''Declaring variables to compute the average accuracy'''
max_accuracy = -1

numOfIter = 1
numOfIterVar = 1
accuracyTrainSum = 0
accuracyTestSum = 0


while numOfIterVar > 0:
    X_train, X_test, y_train, y_test = train_test_split(data, outcome, random_state=numOfIterVar, stratify=outcome, test_size=0.25)

    '''Principal Component Analysis'''
    # pca = PCA(n_components = 7) 
    # X_train = pca.fit_transform(X_train) 
    # X_test = pca.transform(X_test) 

    '''Outlier detection using Mahalanobis Distance'''
    outliers_indices = MD_detectOutliers(X_train)

    print("Outliers Indices: {}\n".format(outliers_indices))
    print("Outliers:")
    print(len(outliers_indices))

    newData = []
    newOutcome = []

    '''Removing the Outlier Indices (rows) from the data'''
    for var in range(len(X_train)):
        if var not in outliers_indices:
            newData.append(X_train[var])
            newOutcome.append(y_train[var])

    newData = np.array(newData)
    newOutcome = np.array(newOutcome)

    '''Logistic Regression'''
    logreg = LogisticRegression().fit(newData, newOutcome)
    train_score = logreg.score(newData, newOutcome)
    test_score = logreg.score(X_test, y_test)
    print("Training set score: {:.3f}".format(train_score))
    print("Test set score: {:.3f}".format(test_score))

    if(test_score > max_accuracy):
        max_accuracy = test_score

    accuracyTrainSum += train_score
    accuracyTestSum += test_score
    numOfIterVar -= 1

    '''Plotting the confusion matrix for the above model'''
    y_pred = logreg.predict(X_test)
    # Plot normalized confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    print(cm)

    # Show confusion matrix in a separate window
    plt.matshow(cm)
    plt.title('Confusion matrix')
    plt.colorbar()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

'''Printing the maximum as well as average accuracy achieved by the model'''
print("Average Train set accuracy: {:.3f}".format(accuracyTrainSum/numOfIter))
print("Average Test set accuracy: {:.3f}".format(accuracyTestSum/numOfIter))
print("Maximum Accuracy: {:.3f}".format(max_accuracy))

print(__doc__)