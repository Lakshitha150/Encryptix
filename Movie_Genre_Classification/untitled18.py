# importing pandas library
import pandas as pd

# read txt file
movie_classification_train = pd.read_csv("Movie_Genre_Classification\train_data.txt", sep=' \:\:\: ', header=None, engine='python')

# adding column headings
movie_classification_train.columns = ['ID','TITLE','GENRE','DESCRIPTION']

# store dataframe into csv file
movie_classification_train.to_csv('Movie_Genre_Classification\train_data.csv',index = None)

# read txt file
movie_classification_test = pd.read_csv("Movie_Genre_Classification\test_data.txt", sep=' \:\:\: ', header=None, engine='python')

# adding column headings
movie_classification_test.columns = ['ID','TITLE','DESCRIPTION']

# store dataframe into csv file
movie_classification_test.to_csv('Movie_Genre_Classification\test_data.csv',index = None)

df1 = pd.read_csv('Movie_Genre_Classification\train_data.csv')
print(df1.shape)
df1.head()

df1['GENRE'].value_counts()

df2 = pd.read_csv('Movie_Genre_Classification\test_data.csv')
print(df2.shape)
df2.head()

df1['genre_num'] = df1['GENRE'].map({
    'drama': 1,
    'documentary': 2,
    'comedy': 3,
    'thriller': 4,
    'horror': 5,
    'short': 6,
    'adult': 7,
    'sci-fi': 8,
    'family': 9,
    'talk-show': 10,
    'action': 11,
    'reality-tv': 12,
    'crime': 13,
    'fantasy': 14,
    'animation': 15,
    'sport': 16,
    'music': 17,
    'adventure': 18,
    'western': 19,
    'mystery': 20,
    'musical': 21,
    'biography': 22,
    'history': 23,
    'game-show': 24,
    'news': 25,
    'war': 26,
    'romance': 27
})

df1.head()



#y_train = df1['genre_num']
#X_train = df1.drop('DESCRIPTION', axis = 1)
#y_test = df2['genre_num']
#X_test = df2.drop('DESCRIPTION', axis = 1)

import sklearn
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df1["DESCRIPTION"], df1['genre_num'], test_size=0.2, random_state=42, stratify=df1['genre_num'])

print(X_train.shape)
print(X_test.shape)

y_train.value_counts()

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import TfidfVectorizer

clf = Pipeline([
    ('vectorizer_tfidf', TfidfVectorizer()),
     ('knn',KNeighborsClassifier())])
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))