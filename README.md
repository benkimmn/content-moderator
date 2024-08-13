# AI-Powered Content Moderator

This project is an AI-powered content moderation system designed to automatically detect and classify offensive language, hate speech, and neutral text in social media posts. The system leverages a Naive Bayes model trained on preprocessed data and provides a RESTful API for real-time content moderation.

## Project Overview

The goal of this project is to create a system that can automatically moderate user-generated content by classifying text into categories such as hate speech, offensive language, and neutral language. The project involves:

1. **Data Preprocessing**: Cleaning and processing the raw text data.
2. **Model Training**: Training a machine learning model to classify text.
3. **API Development**: Creating an API to serve the model for real-time text classification.

## Tech Stack

- **Python**: The core programming language used for data processing, model training, and API development.
- **Flask**: A lightweight web framework used to create the API.
- **scikit-learn**: A machine learning library used for training the Naive Bayes model.
- **Pandas**: A data manipulation library used for data preprocessing.
- **NLTK**: The Natural Language Toolkit used for text preprocessing.

## Dataset

The dataset used for this project is a collection of social media posts from kaggle labeled for hate speech, offensive language, and neutral text. It is preprocessed to remove noise and convert the text into a format suitable for machine learning.

## Project Structure

content_moderator/
│
├── content_moderator_env/ # Virtual environment for the project
├── app.py # Flask API code
├── labeled_data.csv # Original dataset (not included in repo)
├── preprocess_data.py # Script for data preprocessing
├── preprocessed_hate_speech_dataset.csv # Preprocessed dataset
├── train_model.py # Script to train the model
├── content_moderator_model.pkl # Saved trained model
├── tfidf_vectorizer.pkl # Saved TF-IDF vectorizer
└── README.md # Project documentation
