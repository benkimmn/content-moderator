import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
import html
from bs4 import BeautifulSoup

# Download NLTK data (if you haven't already)
nltk.download('stopwords')
nltk.download('punkt')

# Load the dataset
df = pd.read_csv('labeled_data.csv')

# Function to preprocess text
def preprocess_text(text):
    # Decode HTML entities
    text = html.unescape(text)
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove HTML tags if any
    text = BeautifulSoup(text, "html.parser").get_text()
    
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Tokenize text
    words = word_tokenize(text)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]
    
    # Rejoin words into a single string
    return ' '.join(words)

# Apply preprocessing to the text column with progress reporting
print("Starting preprocessing...")
for i, tweet in enumerate(df['tweet']):
    df.at[i, 'processed_text'] = preprocess_text(tweet)
    if i % 100 == 0:
        print(f'Processed {i}/{len(df)} rows')
print("Preprocessing completed.")

# Save the preprocessed data to a new CSV
df.to_csv('preprocessed_hate_speech_dataset.csv', index=False)
