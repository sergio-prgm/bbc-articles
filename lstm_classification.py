# -*- coding: utf-8 -*-
"""
Sentiment Analysis Script
"""

import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from sklearn.manifold import TSNE
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import confusion_matrix
import kagglehub

# -------------------------------
# Dataset Download and Loading
# -------------------------------
# Download latest version of the dataset
path = kagglehub.dataset_download("amunsentom/article-dataset-2")
print("Path to dataset files:", path)

# Look for CSV files in the downloaded folder
csv_files = [f for f in os.listdir(path) if f.endswith('.csv')]
if csv_files:
    dataset_path = os.path.join(path, csv_files[0])
    df = pd.read_csv(dataset_path)
    print(df.head())
else:
    raise FileNotFoundError("No CSV files found in the downloaded folder.")

# -------------------------------
# Sentiment Labeling using VADER
# -------------------------------
nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()

def vader_sentiment_label(text):
    scores = sia.polarity_scores(text)
    compound = scores['compound']
    if compound >= 0.1:
        return 'positive'
    elif compound <= -0.1:
        return 'negative'
    else:
        return 'neutral'

# Apply VADER-based sentiment labeling
df['sentiment'] = df['text'].apply(vader_sentiment_label)
print("Sentiment distribution (VADER):")
print(df['sentiment'].value_counts())

# -------------------------------
# Use Original Dataset & Compute Class Weights
# -------------------------------
# Use the entire dataset rather than aggressively balancing it.
X = df['text'].astype(str).values
sentiment_map = {'positive': 0, 'neutral': 1, 'negative': 2}
Y = np.array([sentiment_map[label] for label in df['sentiment']])

# Compute class weights to avoid imbalance
classes = np.unique(Y)
class_weights_array = compute_class_weight('balanced', classes=classes, y=Y)
class_weights = {i: weight for i, weight in enumerate(class_weights_array)}
print("Computed class weights:", class_weights)

# -------------------------------
# Tokenization and Padding
# -------------------------------
max_words = 20000  # Maximum vocabulary size
tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(X)
X_seq = tokenizer.texts_to_sequences(X)

# Use a fixed maximum sequence length
maxlen = 300
X_pad = pad_sequences(X_seq, maxlen=maxlen, padding='post', truncating='post')

# Convert labels to one-hot encoding
Y_categorical = to_categorical(Y, num_classes=3)

# -------------------------------
# Load Pre-trained GloVe Embeddings and Build Embedding Matrix
# -------------------------------
embedding_dim = 100
glove_path = 'glove.6B.100d.txt'
embeddings_index = {}

with open(glove_path, 'r', encoding='utf8') as f:
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs

word_index = tokenizer.word_index
embedding_matrix = np.zeros((len(word_index) + 1, embedding_dim))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

# -------------------------------
# Word Embeddings Visualization using t-SNE
# -------------------------------
words = list(embeddings_index.keys())[:100]  # Pick first 100 words
word_vectors = np.array([embeddings_index[w] for w in words])

tsne = TSNE(n_components=2, random_state=42)
word_vec_2d = tsne.fit_transform(word_vectors)

plt.figure(figsize=(7, 7))
plt.scatter(word_vec_2d[:, 0], word_vec_2d[:, 1], marker='.')
for i, word in enumerate(words):
    plt.annotate(word, (word_vec_2d[i, 0], word_vec_2d[i, 1]), fontsize=9)
plt.title('Word Embeddings Visualization (t-SNE)')
plt.show()

# -------------------------------
# Define the LSTM Model
# -------------------------------
model = Sequential([
    Embedding(input_dim=len(word_index) + 1,
              output_dim=embedding_dim,
              weights=[embedding_matrix],
              input_length=maxlen,
              trainable=False),
    Bidirectional(LSTM(units=64, return_sequences=True)),
    Dropout(0.3),
    Bidirectional(LSTM(units=32)),
    Dropout(0.3),
    Dense(3, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# -------------------------------
# Train the Model with Class Weights
# -------------------------------
epochs = 50
lr_reduce = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)

history = model.fit(X_pad, Y_categorical, epochs=epochs, batch_size=32,
                    validation_split=0.2, class_weight=class_weights, callbacks=[lr_reduce])

# -------------------------------
# Plot Training History
# -------------------------------
plt.figure(figsize=(12, 5))

# Plot Loss
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Model Loss')

# Plot Accuracy
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Model Accuracy')
plt.show()

# -------------------------------
# Evaluate the Model and Show Confusion Matrix
# -------------------------------
y_pred = model.predict(X_pad)
y_pred_classes = y_pred.argmax(axis=1)
y_true = Y_categorical.argmax(axis=1)

cm = confusion_matrix(y_true, y_pred_classes)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Positive', 'Neutral', 'Negative'],
            yticklabels=['Positive', 'Neutral', 'Negative'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()


# -------------------------------
# Define a Function to Predict Sentiment for New Articles
# -------------------------------
def predict_sentiment(text):
    seq = tokenizer.texts_to_sequences([text])
    pad_seq = pad_sequences(seq, maxlen=maxlen, padding='post', truncating='post')
    prediction = model.predict(pad_seq)
    sentiment = np.argmax(prediction)
    reverse_map = {0: 'positive', 1: 'neutral', 2: 'negative'}
    return reverse_map[sentiment]

# Example texts to classify
test_texts = [
    # Positive
    "The company reported a significant increase in profits this quarter.",
    "Local communities celebrate after a successful campaign to protect a vital nature reserve from development.",
    "A new technology breakthrough promises to revolutionize the way we generate clean energy, reducing carbon emissions significantly.",
    "The city has been recognized for its commitment to sustainability, with a new green initiative aimed at reducing the city's carbon footprint.",
    "A major pharmaceutical company has developed a new treatment for a rare disease, offering hope to thousands of patients worldwide.",
    
    # Neutral
    "Researchers are investigating the potential impact of a new drug on human health, with initial tests showing mixed results.",
    "The city council is planning to hold a public consultation regarding proposed changes to local zoning laws.",

    # Negative
    "Protests erupted across the city following the controversial decision to raise fuel prices by 20%.",
    "The economic slowdown is causing widespread job losses, with many small businesses struggling to stay afloat.",
    "The recent rise in global temperatures is causing unprecedented wildfires across continents, threatening biodiversity and human lives.",
    "A large-scale cyberattack has compromised critical infrastructure in several countries, paralyzing government services and causing widespread panic.",
    "A historic drought has devastated agricultural output in multiple regions, leading to severe food shortages and skyrocketing prices."
    
]

for i, text in enumerate(test_texts):
    sentiment = predict_sentiment(text)
    print(f"Test {i+1}: {text}\nPredicted Sentiment: {sentiment}\n")

