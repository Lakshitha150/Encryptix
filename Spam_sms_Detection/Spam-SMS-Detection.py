import numpy as np
import pandas as pd
import csv
import chardet
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.svm import SVC  # For Support Vector Classification
from sklearn.metrics import accuracy_score,confusion_matrix,precision_score

file_path = 'Spam_sms_Detection/spam.csv'

with open(file_path, 'rb') as file:
  rawdata = file.read()
  result = chardet.detect(rawdata)
  encoding = result['encoding']

with open(file_path, 'r', encoding=encoding) as file:
  reader = csv.reader(file)
  data = list(reader)
  print('Encoding of CSV file: ',encoding)

df=pd.read_csv(file_path,encoding=encoding)

df.head()

df.shape

df.info()

#Data Cleaning

df.drop(columns=['Unnamed: 2','Unnamed: 3','Unnamed: 4'],inplace=True)


df.rename(columns={'v1':'target','v2':'text'},inplace=True)

from sklearn.preprocessing import LabelEncoder
encoder=LabelEncoder()
df['target']= encoder.fit_transform(df['target'])

df.head()

df.isnull().sum()

df.duplicated().sum()

df.drop_duplicates(keep='first',inplace=True)

df.head()

import matplotlib.pyplot as plt
plt.pie(df['target'].value_counts(),labels=['ham','spam'],autopct="%0.2f")
plt.show()


nltk.download('punkt')

#Data Processing


ps=PorterStemmer()
nltk.download('stopwords')
import string


def transform_text(text):
  text = text.lower()
  text = nltk.word_tokenize(text)


  y = []
  for i in text:
    if i.isalnum():
      y.append(i)
  text = y[:]
  y.clear()

  for i in text:
    if i not in stopwords.words('english') and i not in string.punctuation:
      y.append(i)

  text = y[:]
  y.clear()

  for i in text:
    y.append(ps.stem(i))

  return " ".join(y)

df['transformed_text']=df['text'].apply(transform_text)

df.head()

#Building Model


cv=CountVectorizer()
tfidf = TfidfVectorizer(max_features=3000)

X = tfidf.fit_transform(df['transformed_text']).toarray()

X.shape

Y= df['target'].values

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=2)


clf = svm.SVC(kernel='linear') # Linear Kernel
clf.fit(X_train,Y_train)
Y_predict = clf.predict(X_test)
print("Support Vector Machine results")
print('Accuracy: ',accuracy_score(Y_test,Y_predict))
print('Confusion Matrix: ')
print(confusion_matrix(Y_test,Y_predict))
print('Precision Score: ',precision_score(Y_test,Y_predict))

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Create a logistic regression model
model = LogisticRegression()

# Train the model
model.fit(X_train, Y_train)

# Make predictions
Y_predict2 = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(Y_test, Y_predict2)
confusion = confusion_matrix(Y_test, Y_predict2)
precision = precision_score(Y_test, Y_predict2)

print("Logistic Regression Results")
print("Accuracy:", accuracy)
print("Confusion Matrix:")
print(confusion)
print("Precision:", precision)