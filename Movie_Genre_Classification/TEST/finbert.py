import pandas as pd
from tqdm import tqdm
import time
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import classification_report, accuracy_score

# Loading training data
start_time = time.time()
movie_classification_train = pd.read_csv("Movie_Genre_Classification/train_data-Copy.txt", sep=' \:\:\: ', header=None, engine='python')
end_time = time.time()
print(f"Loading train data took {end_time - start_time:.2f} seconds.")

movie_classification_train.columns = ['ID', 'TITLE', 'GENRE', 'DESCRIPTION']
movie_classification_train.to_csv('Movie_Genre_Classification/train_data-Copy.csv', index=None)

# Loading test data
start_time = time.time()
movie_classification_test = pd.read_csv("Movie_Genre_Classification/test_data-Copy.txt", sep=' \:\:\: ', header=None, engine='python')
end_time = time.time()
print(f"Loading test data took {end_time - start_time:.2f} seconds.")

movie_classification_test.columns = ['ID', 'TITLE', 'DESCRIPTION']
movie_classification_test.to_csv('Movie_Genre_Classification/test_data-Copy.csv', index=None)

# Loading test solution data
start_time = time.time()
test_solutions = pd.read_csv("Movie_Genre_Classification/train_data-Copy.txt", sep=' \:\:\: ', header=None, engine='python')
end_time = time.time()
print(f"Loading test solution data took {end_time - start_time:.2f} seconds.")

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

# Load FinBERT model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")

# Function to classify texts using FinBERT
def classify_texts(texts):
    pipe = pipeline("text-classification", model=model, tokenizer=tokenizer, return_all_scores=True)
    results = pipe(texts)
    return [result[0]['label'] for result in results]  # Assuming single label classification

# Using tqdm to display the progress bar during prediction
with tqdm(total=1, desc="Predicting the test data") as pbar:
    y_pred = classify_texts(X_test.tolist())
    pbar.update(1)

# Mapping predictions back to numerical values (assuming FinBERT provides genre names)
genre_inverse_mapping = {v: k for k, v in genre_mapping.items()}
y_pred_num = [genre_inverse_mapping[pred] for pred in y_pred]

# Evaluating the model's performance
start_time = time.time()
print(classification_report(y_test, y_pred_num))
print("Accuracy:", accuracy_score(y_test, y_pred_num))
end_time = time.time()
print(f"Evaluating the model took {end_time - start_time:.2f} seconds.")

# Storing the predictions to a CSV file
df2['predicted_genre_num'] = y_pred_num
df2['predicted_genre'] = df2['predicted_genre_num'].map(genre_inverse_mapping)

df2.to_csv('Movie_Genre_Classification/test_data_with_predictions-Copy.csv', index=None)

# Displaying the results
print(df2.head())
