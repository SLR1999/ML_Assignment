import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA 
from confusionMatrix import plot_confusion_matrix
from sklearn.preprocessing import StandardScaler

'''The point above fence_low and below fence_high will be removed as they are considered outliers'''
def remove_outlier(df_in, col_name):
    q1 = df_in[col_name].quantile(0.25)
    q3 = df_in[col_name].quantile(0.75)
    iqr = q3-q1 #Interquartile range
    fence_low  = q1-1.5*iqr
    fence_high = q3+1.5*iqr
    df_out = df_in.loc[(df_in[col_name] > fence_low) & (df_in[col_name] < fence_high)]
    return df_out   

'''Reading csv file into a dataframe named diabetes'''
diabetes = pd.read_csv('Pima_Indian_diabetes.csv')

# '''Plotting the heatmap of correlation matrix'''
# sns.heatmap(diabetes.corr(),cmap='Blues',annot=False)

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

'''lremove list has columns that should be removed from diabetes set a they have value as NaN's'''
lremove = []
'''lrequired list has valid columns'''
lrequired = []
for i in range(0, len(columnNames)):
    colName = columnNames[i]
    # if (colName == "Glucose" or colName == "BMI" or colName == "BloodPressure"):
    #     diabetes[colName] = diabetes[colName].replace(0, np.NaN)
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
    lremove.append(colName)
    lremove.append("Modified" + colName)
    lrequired.append("Final" + colName)

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

'''Removes invalid columns in diabetes and assigns it to data'''
data = diabetes.drop(lremove, axis = 1)

'''Here the outliers are removed column wise by calling the remove outlier function'''
for var in range(len(columnNames)):
    remove_outlier(data, "Final" + columnNames[var])

# df = pd.DataFrame(np.random.randn(10,8), columns=lrequired)
# boxplot = df.boxplot(column=lrequired)
# sns.boxplot(x="diagnosis", y="area_mean", data=data)
# # data.boxplot(column = 'area_mean', by = 'diagnosis')
# plt.title('')


'''Standardizing the data using standardscaler function'''
x = data.loc[:, lrequired].values
y = data.loc[:,['Outcome']].values
x = StandardScaler().fit_transform(x)

'''Declaring variables to compute the average accuracy'''
max_accuracy = -1

numOfIter = 1
numOfIterVar = 1
accuracyTrainSum = 0
accuracyTestSum = 0

'''Principal Component Analysis'''
pca = PCA(n_components=8)
principalComponentsDiabetes = pca.fit_transform(x)
principalDf = pd.DataFrame(data = principalComponentsDiabetes, columns = ['principal component 1', 'principal component 2', 'principal component 3', 'principal component 4', 'principal component 5', 'principal component 6', 'principal component 7', 'principal component 8'])


while numOfIterVar > 0: 
    print ("IN THE LOOP")
    X_train, X_test, y_train, y_test = train_test_split(principalDf, y, random_state=5932, test_size=0.25)

    # '''Logistic Regression'''
    logreg = LogisticRegression().fit(X_train, y_train)
    train_score = logreg.score(X_train, y_train)
    test_score = logreg.score(X_test, y_test)
    print("Training set score: {:.3f}".format(train_score))
    print("Test set score: {:.3f}".format(test_score))

    if(test_score > max_accuracy):
        max_accuracy = test_score
        random_max_value = numOfIterVar

    accuracyTrainSum += train_score
    accuracyTestSum += test_score
    numOfIterVar -= 1

    '''Plotting the confusion matrix for the above model'''
    y_pred = logreg.predict(X_test)
    # Plot normalized confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    # print(cm)

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
print("Max Accuracy random value: {:.3f}" .format(random_max_value))
print("Maximum Accuracy: {:.3f}".format(max_accuracy))
