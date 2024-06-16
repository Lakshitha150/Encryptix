import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

#define path
Test_path = 'fraudTest.csv'
Train_path = 'fraudTrain.csv'
result_path = 'Result_Report/Report.txt'

#by changing frac value you can reduce the Test data size
Test_sample = pd.read_csv(Test_path).sample(frac = 1)
Train_sample = pd.read_csv(Train_path).sample(frac = 1)

# check details
Test_sample.head()
Test_sample.info()

#define non numeric columns
non_numeric_columns = Train_sample.select_dtypes(include=['object']).columns
non_numeric_columns

# Create a copy of Test_sample
Train_sample_num = Train_sample.copy()
Test_sample_num = Test_sample.copy()

#drop all object columns
Train_sample_num = Train_sample_num.select_dtypes(exclude=['object'])
Test_sample_num = Test_sample_num.select_dtypes(exclude=['object'])


Train_sample_num.head()
Test_sample_num.head()

#Distribution of No Fraud vs Fraud
Train_sample_num["is_fraud"].value_counts()

#This dataset is Higly Unbalance

#separationg data for analysis
legit = Train_sample_num[Train_sample.is_fraud == 0]
fraud = Train_sample_num[Train_sample.is_fraud == 1]

print (legit.shape)
print (fraud.shape)

#stratistical measures for data
legit.amt.describe()

fraud.amt.describe()

#compare the value for both transfaction
Train_sample_num.groupby('is_fraud').mean()

#Under Sampling and create new dataset

print ("fraudalant tranfaction :"+str(fraud.shape[0]))

#randomly select legimate transaction data equal to fraudalant data present in dataset
legit_sample = legit.sample(n=fraud.shape[0])

#concatenating two dataset

New_train_dataset = pd.concat([legit_sample,fraud], axis = 0)

New_train_dataset.head()

New_train_dataset["is_fraud"].value_counts()

New_train_dataset.select_dtypes(include=['number']).groupby('is_fraud').mean()

X_train= New_train_dataset.drop(columns = 'is_fraud', axis = 1)
Y_train = New_train_dataset['is_fraud']

X_test = Test_sample_num.drop(columns = 'is_fraud', axis = 1)
Y_test = Test_sample_num['is_fraud']

model = DecisionTreeClassifier()
model.fit(X_train, Y_train)

Decision_prediction = model.predict(X_test)

Accuracy=accuracy_score(Y_test, Decision_prediction)
Con_matrix=confusion_matrix(Y_test, Decision_prediction)
report=classification_report(Y_test, Decision_prediction)
print("Accuracy of test data for Decition Tree Regressor : ",Accuracy )
print("Confution Matrix Score : ",Con_matrix)
print("Report : ",report)

# Create Text File with Report and Accuracy
with open(result_path,'w') as f:
    f.write("Classification Report:\n")
    f.write(str(report))
    f.write("\n")
    f.write(str(Accuracy))
