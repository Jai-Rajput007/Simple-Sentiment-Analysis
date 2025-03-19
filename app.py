import streamlit as st
import torch
import numpy as np
import re

# Load pre-trained GloVe embeddings
embedding_dim = 300
embeddings_index = {}
with open('glove.6B.300d.txt', encoding='utf-8') as f:
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs

# Function to preprocess text and convert to embeddings
def text_to_embedding(text, embeddings_index, embedding_dim):
    text = re.sub(r'[^\w\s]', '', text.lower())
    words = text.split()
    embeddings = [embeddings_index[word] for word in words if word in embeddings_index]
    if not embeddings:
        return np.zeros(embedding_dim)
    embeddings = np.array(embeddings)
    return np.mean(embeddings, axis=0)

# Define the SentimentClassifier model
class SentimentClassifier(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(SentimentClassifier, self).__init__()
        self.fc1 = torch.nn.Linear(input_dim, hidden_dim)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(hidden_dim, 1)
        self.sigmoid = torch.nn.Sigmoid()
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x

# Load the saved model
model = SentimentClassifier(embedding_dim, 128)
model.load_state_dict(torch.load('sentiment_classifier.pth', map_location=torch.device('cpu')))
model.eval()

# Streamlit app
st.title("Sentiment Analysis")

# Text input
text = st.text_input("Enter your text here:")

# Predict sentiment
if text:
    embedding = text_to_embedding(text, embeddings_index, embedding_dim)
    embedding_tensor = torch.tensor(embedding, dtype=torch.float32).view(1, -1)
    with torch.no_grad():
        output = model(embedding_tensor)
        prob = output.item()
        sentiment = "Positive" if prob >= 0.5 else "Negative"
    st.write(f"Sentiment: {sentiment}, Probability: {prob:.4f}")