from flask import Flask, request, jsonify
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

app = Flask(__name__)

# Load the trained model and TF-IDF vectorizer
with open('content_moderator_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('tfidf_vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

@app.route('/moderate', methods=['POST'])
def moderate():
    data = request.get_json()
    text = data.get('text', '')

    # Preprocess and vectorize the input text
    text_tfidf = vectorizer.transform([text])
    prediction = model.predict(text_tfidf)[0]

    return jsonify({'classification': int(prediction)})

if __name__ == '__main__':
    app.run(debug=True)
