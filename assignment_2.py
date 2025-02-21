# -*- coding: utf-8 -*-
"""
Improved Sentiment Analysis Script with Less Aggressive Balancing
"""

import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

# Ensure GPU is available if you have one
print("GPU Available:", tf.config.list_physical_devices('GPU'))

# -------------------------------
# Dataset Download and Loading
# -------------------------------
import kagglehub

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
import nltk
nltk.download('vader_lexicon')
from nltk.sentiment import SentimentIntensityAnalyzer

sia = SentimentIntensityAnalyzer()

def vader_sentiment_label(text):
    scores = sia.polarity_scores(text)
    compound = scores['compound']
    if compound >= 0.05:
        return 'positive'
    elif compound <= -0.05:
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

# Compute class weights to help with imbalance
from sklearn.utils.class_weight import compute_class_weight
classes = np.unique(Y)
class_weights_array = compute_class_weight('balanced', classes=classes, y=Y)
class_weights = {i: weight for i, weight in enumerate(class_weights_array)}
print("Computed class weights:", class_weights)

# -------------------------------
# Tokenization and Padding
# -------------------------------
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

max_words = 20000  # Maximum vocabulary size
tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(X)
X_seq = tokenizer.texts_to_sequences(X)

# Use a fixed maximum sequence length (e.g., 300 tokens)
maxlen = 300
X_pad = pad_sequences(X_seq, maxlen=maxlen, padding='post', truncating='post')

# Convert labels to one-hot encoding
Y_categorical = to_categorical(Y, num_classes=3)

# -------------------------------
# Load Pre-trained GloVe Embeddings and Build Embedding Matrix
# -------------------------------
embedding_dim = 100
glove_path = 'glove.6B.100d.txt'  # Ensure this file is in your working directory
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
# Define the Improved LSTM Model
# -------------------------------
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping

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
# Train the Model with Early Stopping & Class Weights
# -------------------------------
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
epochs = 50  # You can adjust the number of epochs as needed

history = model.fit(X_pad, Y_categorical, epochs=epochs, batch_size=32,
                    validation_split=0.2, class_weight=class_weights, callbacks=[early_stop])

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
from sklearn.metrics import confusion_matrix

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

# Example predictions
test_texts = [
    "The company reported a significant increase in profits this quarter.",
    "The local hospital is facing severe budget cuts, leading to staff layoffs.",
    "The government is reviewing its policies on immigration in an effort to improve security."
]

for i, text in enumerate(test_texts):
    sentiment = predict_sentiment(text)
    print(f"Test {i+1}: {text}\nPredicted Sentiment: {sentiment}\n")
