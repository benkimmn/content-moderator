import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import pickle

print("Loading dataset...")
df = pd.read_csv('preprocessed_hate_speech_dataset.csv')
print("Dataset loaded.")

X = df['processed_text']
y = df['class']  # Assuming 'class' is the label column

print("Splitting dataset...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Dataset split into training and test sets.")

print("Vectorizing text data...")
vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)
print("Text data vectorized.")

print("Training the model...")
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)
print("Model trained.")

y_pred = model.predict(X_test_tfidf)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')
print(classification_report(y_test, y_pred))

# Save the model and vectorizer
print("Saving the model and vectorizer...")
with open('content_moderator_model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('tfidf_vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

print("Model and vectorizer saved.")
