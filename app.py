import streamlit as st
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from datasets import load_dataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import re
from collections import defaultdict
import requests
import os

# Function to download the GloVe file if it doesn't exist
def download_glove_file():
    glove_url = "https://drive.google.com/uc?export=download&id=1aBcDeFgHiJkLmNoPqRsTuVwXyZ"  # Replace with your URL
    glove_path = "glove.6B.50d.txt"
    if not os.path.exists(glove_path):
        with st.spinner("Downloading GloVe embeddings..."):
            response = requests.get(glove_url)
            with open(glove_path, "wb") as f:
                f.write(response.content)
        st.success("GloVe embeddings downloaded successfully!")

# Function to handle contractions
def expand_contractions(text):
    contractions = {
        "i'm": "i am",
        "it's": "it is",
        "don't": "do not",
        "won't": "will not",
        "can't": "cannot",
        "i've": "i have",
        "you're": "you are",
        "he's": "he is",
        "she's": "she is",
        "we're": "we are",
        "they're": "they are"
    }
    text = text.lower()
    for contraction, expanded in contractions.items():
        text = text.replace(contraction, expanded)
    return text

# Function to preprocess text and convert to embeddings
@st.cache_data
def text_to_embedding(text, embeddings_index, embedding_dim):
    text = expand_contractions(text)
    text = re.sub(r'[^\w\s]', '', text)
    words = text.split()
    embeddings = []
    for word in words:
        if word in embeddings_index:
            embeddings.append(embeddings_index[word])
    if not embeddings:
        return np.zeros(embedding_dim)
    embeddings = np.array(embeddings)
    return np.mean(embeddings, axis=0)

# Define the neural network for sentiment classification
class SentimentClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(SentimentClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x

# Function to predict sentiment on new text
def predict_sentiment(text, model, embeddings_index, embedding_dim):
    embedding = text_to_embedding(text, embeddings_index, embedding_dim)
    embedding_tensor = torch.tensor(embedding, dtype=torch.float32).view(1, -1)
    model.eval()
    with torch.no_grad():
        output = model(embedding_tensor)
        prob = output.item()
        pred = 1 if prob >= 0.5 else 0
        sentiment = "Positive" if pred == 1 else "Negative"
    return sentiment, prob

# Load GloVe embeddings and model (cached in session state)
if 'embeddings_index' not in st.session_state or 'model' not in st.session_state:
    with st.spinner("Loading GloVe embeddings and model..."):
        # Download GloVe file if it doesn't exist
        download_glove_file()

        # Load GloVe embeddings
        embedding_dim = 50
        embeddings_index = {}
        with open('glove.6B.50d.txt', encoding='utf-8') as f:
            for line in f:
                values = line.split()
                word = values[0]
                coefs = np.asarray(values[1:], dtype='float32')
                embeddings_index[word] = coefs
        st.session_state.embeddings_index = embeddings_index

        # Load the pre-trained model
        input_dim = embedding_dim
        hidden_dim = 128
        model = SentimentClassifier(input_dim, hidden_dim)
        model.load_state_dict(torch.load('model.pth'))
        model.eval()
        st.session_state.model = model

        # Load test set for metrics (optional, can be precomputed)
        imdb_dataset = load_dataset("imdb")
        amazon_dataset = load_dataset("amazon_polarity")
        imdb_test_df = pd.DataFrame({'text': imdb_dataset['test']['text'], 'label': imdb_dataset['test']['label']})
        amazon_test_df = pd.DataFrame({'text': amazon_dataset['test']['content'], 'label': amazon_dataset['test']['label']})
        test_df = pd.concat([
            imdb_test_df.sample(1000, random_state=42),
            amazon_test_df.sample(1000, random_state=42)
        ], ignore_index=True)
        X_test = np.array([text_to_embedding(text, embeddings_index, embedding_dim) for text in test_df['text']])
        y_test = test_df['label'].values
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
        y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

        model.eval()
        with torch.no_grad():
            test_outputs = model(X_test_tensor)
            test_preds = (test_outputs >= 0.5).float().numpy()
        accuracy = accuracy_score(y_test, test_preds)
        precision = precision_score(y_test, test_preds)
        recall = recall_score(y_test, test_preds)
        f1 = f1_score(y_test, test_preds)
        st.session_state.test_metrics = (accuracy, precision, recall, f1)

# Streamlit app
st.title("Sentiment Analysis with GloVe Embeddings")
st.write("This app analyzes the sentiment of any text (positive or negative) using a neural network trained on GloVe embeddings.")

# Display test set metrics
accuracy, precision, recall, f1 = st.session_state.test_metrics
st.write("### Model Performance on Test Set")
st.write(f"**Accuracy**: {accuracy:.4f}")
st.write(f"**Precision**: {precision:.4f}")
st.write(f"**Recall**: {recall:.4f}")
st.write(f"**F1-score**: {f1:.4f}")

# Test on example texts
st.write("### Example Texts")
example_texts = [
    "I love this movie, it’s amazing!",
    "This phone is terrible, it keeps crashing.",
    "I had an amazing day at the park with my friends!",
    "The lecture was boring and unhelpful.",
    "I’m so excited for the weekend, it’s going to be great!",
    "The food at this restaurant was disappointing and overpriced.",
    "I really enjoyed the concert last night, the music was fantastic!",
    "My new laptop is super fast and easy to use.",
    "The weather today is awful, I hate this rain!",
    "I’m feeling so happy after talking to my best friend."
]

for text in example_texts:
    sentiment, prob = predict_sentiment(text, st.session_state.model, st.session_state.embeddings_index, embedding_dim)
    st.write(f"**Text**: {text}")
    st.write(f"**Sentiment**: {sentiment}, **Probability**: {prob:.4f}")
    st.write("---")

# Interactive sentiment prediction
st.write("### Try Your Own Text")
user_input = st.text_input("Enter a text to predict its sentiment:", "I'm having dinner of human parts")
if user_input:
    with st.spinner("Analyzing sentiment..."):
        sentiment, prob = predict_sentiment(user_input, st.session_state.model, st.session_state.embeddings_index, embedding_dim)
        st.write(f"**Sentiment**: {sentiment}")
        st.write(f"**Probability**: {prob:.4f}")
