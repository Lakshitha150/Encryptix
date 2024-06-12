# Importing necessary libraries
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer

# Define file paths
train_data_path = "Movie_Genre_Classification/train_data-sample.txt"
train_data_csv_path = "Movie_Genre_Classification/train_data-sample.csv"
test_data_path = "Movie_Genre_Classification/test_data-sample.txt"
test_data_csv_path = "Movie_Genre_Classification/test_data-sample.csv"
test_solutions_path = "Movie_Genre_Classification/test_data_solution-sample.txt"
output_predictions_path = "Movie_Genre_Classification/test_data_with_predictions-sample.csv"
original_vs_predicted_path = "Movie_Genre_Classification/RESULT_CSV/KNN_original_vs_predicted-sample.csv"
result_path = "Movie_Genre_Classification/RESULT_CSV/KNN_Result.txt"

# Reading training data from the text file
movie_classification_train = pd.read_csv(train_data_path, sep=' \:\:\: ', header=None, engine='python')

# Adding column headings
movie_classification_train.columns = ['ID', 'TITLE', 'GENRE', 'DESCRIPTION']

# Storing the dataframe into a CSV file
movie_classification_train.to_csv(train_data_csv_path, index=None)

# Reading test data from the text file
movie_classification_test = pd.read_csv(test_data_path, sep=' \:\:\: ', header=None, engine='python')

# Adding column headings
movie_classification_test.columns = ['ID', 'TITLE', 'DESCRIPTION']

# Storing the dataframe into a CSV file
movie_classification_test.to_csv(test_data_csv_path, index=None)

# Reading test solution data from the text file
test_solutions = pd.read_csv(test_solutions_path, sep=' \:\:\: ', header=None, engine='python')

# Adding column headings
test_solutions.columns = ['ID', 'TITLE', 'GENRE', 'DESCRIPTION']

# Reading the stored CSV files to verify the data
df1 = pd.read_csv(train_data_csv_path)
print(df1.shape)
print(df1.head())

# Checking the distribution of genres
print(df1['GENRE'].value_counts())

df2 = pd.read_csv(test_data_csv_path)
print(df2.shape)
print(df2.head())

# Mapping genres to numerical values
genre_mapping = {
    'drama': 1, 'documentary': 2, 'comedy': 3, 'thriller': 4, 'horror': 5,
    'short': 6, 'adult': 7, 'sci-fi': 8, 'family': 9, 'talk-show': 10,
    'action': 11, 'reality-tv': 12, 'crime': 13, 'fantasy': 14,
    'animation': 15, 'sport': 16, 'music': 17, 'adventure': 18,
    'western': 19, 'mystery': 20, 'musical': 21, 'biography': 22,
    'history': 23, 'game-show': 24, 'news': 25, 'war': 26, 'romance': 27
}
df1['genre_num'] = df1['GENRE'].map(genre_mapping)
test_solutions['genre_num'] = test_solutions['GENRE'].map(genre_mapping)

print(df1.head())
print(test_solutions.head())

# Combining TITLE and DESCRIPTION for training and test datasets
df1['TEXT'] = df1['TITLE'] + " " + df1['DESCRIPTION']
df2['TEXT'] = df2['TITLE'] + " " + df2['DESCRIPTION']

# Preparing the train and test datasets
X_train = df1["TEXT"]
y_train = df1['genre_num']

X_test = df2["TEXT"]
y_test = test_solutions['genre_num']

# Ensuring the lengths of y_test and y_pred match by slicing test_solutions to match the length of df2
test_solutions = test_solutions.loc[test_solutions['ID'].isin(df2['ID'])]

# Building and training the classification model
clf = Pipeline([
    ('vectorizer_tfidf', TfidfVectorizer()),
    ('knn', KNeighborsClassifier())
])
clf.fit(X_train, y_train)

# Making predictions on the test data
y_pred = clf.predict(X_test)

# Evaluating the model's performance
report=classification_report(y_test, y_pred, zero_division=0)
accuracy=f"Accuracy Score: {accuracy_score(y_test,y_pred)}\n"
print(report)
print(accuracy)
# Create Text File with Report and Accuracy
with open(result_path,'w') as f:
    f.write("Classification Report:\n")
    f.write(report)
    f.write("\n")
    f.write(accuracy)

# Storing the predictions to a CSV file
df2['predicted_genre_num'] = y_pred

# Inverting the genre mapping for easier readability of the results
inverse_genre_mapping = {v: k for k, v in genre_mapping.items()}
df2['predicted_genre'] = df2['predicted_genre_num'].map(inverse_genre_mapping)

df2.to_csv(output_predictions_path, index=None)

# Creating a new dataframe for title, original genre, and predicted genre
original_vs_predicted = df2[['TITLE']].copy()
original_vs_predicted['original_genre'] = test_solutions['GENRE'].reset_index(drop=True)
original_vs_predicted['predicted_genre'] = df2['predicted_genre']

# Storing the original vs predicted genres to a new CSV file
original_vs_predicted.to_csv(original_vs_predicted_path, index=None)

# Displaying the results
print(original_vs_predicted.head())