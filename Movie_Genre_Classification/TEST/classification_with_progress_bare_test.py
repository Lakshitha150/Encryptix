import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm
import time

# Reading training data from the text file
start_time = time.time()
movie_classification_train = pd.read_csv("Movie_Genre_Classification/train_data-Copy.txt", sep=' \:\:\: ', header=None, engine='python')
end_time = time.time()
print(f"Loading train data took {end_time - start_time:.2f} seconds.")

# Adding column headings
movie_classification_train.columns = ['ID', 'TITLE', 'GENRE', 'DESCRIPTION']

# Storing the dataframe into a CSV file
movie_classification_train.to_csv('Movie_Genre_Classification/train_data-Copy.csv', index=None)

# Reading test data from the text file
start_time = time.time()
movie_classification_test = pd.read_csv("Movie_Genre_Classification/test_data-Copy.txt", sep=' \:\:\: ', header=None, engine='python')
end_time = time.time()
print(f"Loading test data took {end_time - start_time:.2f} seconds.")

# Adding column headings
movie_classification_test.columns = ['ID', 'TITLE', 'DESCRIPTION']

# Storing the dataframe into a CSV file
movie_classification_test.to_csv('Movie_Genre_Classification/test_data-Copy.csv', index=None)

# Reading test solution data from the text file
start_time = time.time()
test_solutions = pd.read_csv("Movie_Genre_Classification/train_data-Copy.txt", sep=' \:\:\: ', header=None, engine='python')
end_time = time.time()
print(f"Loading test solution data took {end_time - start_time:.2f} seconds.")

# Adding column headings
test_solutions.columns = ['ID', 'TITLE', 'GENRE', 'DESCRIPTION']

# Reading the stored CSV files to verify the data
start_time = time.time()
df1 = pd.read_csv('Movie_Genre_Classification/train_data-Copy.csv')
print(df1.shape)
print(df1.head())
end_time = time.time()
print(f"Loading train CSV file took {end_time - start_time:.2f} seconds.")

# Checking the distribution of genres
print(df1['GENRE'].value_counts())

start_time = time.time()
df2 = pd.read_csv('Movie_Genre_Classification/test_data-Copy.csv')
print(df2.shape)
print(df2.head())
end_time = time.time()
print(f"Loading test CSV file took {end_time - start_time:.2f} seconds.")

# Mapping genres to numerical values
genre_mapping = {
    'drama': 1, 'documentary': 2, 'comedy': 3, 'thriller': 4, 'horror': 5,
    'short': 6, 'adult': 7, 'sci-fi': 8, 'family': 9, 'talk-show': 10,
    'action': 11, 'reality-tv': 12, 'crime': 13, 'fantasy': 14,
    'animation': 15, 'sport': 16, 'music': 17, 'adventure': 18,
    'western': 19, 'mystery': 20, 'musical': 21, 'biography': 22,
    'history': 23, 'game-show': 24, 'news': 25, 'war': 26, 'romance': 27
}
start_time = time.time()
df1['genre_num'] = df1['GENRE'].map(genre_mapping)
test_solutions['genre_num'] = test_solutions['GENRE'].map(genre_mapping)
end_time = time.time()
print(f"Mapping genres took {end_time - start_time:.2f} seconds.")

print(df1.head())
print(test_solutions.head())

# Preparing the train and test datasets
X_train = df1["DESCRIPTION"]
y_train = df1['genre_num']

X_test = df2["DESCRIPTION"]
y_test = test_solutions['genre_num']

# Building and training the classification model
start_time = time.time()
clf = Pipeline([
    ('vectorizer_tfidf', TfidfVectorizer()),
    ('knn', KNeighborsClassifier())
])

# Using tqdm to display the progress bar during training
with tqdm(total=1, desc="Training the model") as pbar:
    clf.fit(X_train, y_train)
    pbar.update(1)

end_time = time.time()
print(f"Training the model took {end_time - start_time:.2f} seconds.")

# Making predictions on the test data
start_time = time.time()
y_pred = clf.predict(X_test)
end_time = time.time()
print(f"Making predictions took {end_time - start_time:.2f} seconds.")

# Evaluating the model's performance
start_time = time.time()
print(classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))
end_time = time.time()
print(f"Evaluating the model took {end_time - start_time:.2f} seconds.")

# Storing the predictions to a CSV file
df2['predicted_genre_num'] = y_pred

# Inverting the genre mapping for easier readability of the results
inverse_genre_mapping = {v: k for k, v in genre_mapping.items()}
df2['predicted_genre'] = df2['predicted_genre_num'].map(inverse_genre_mapping)

df2.to_csv('Movie_Genre_Classification/test_data_with_predictions-Copy.csv', index=None)

# Displaying the results
print(df2.head())
