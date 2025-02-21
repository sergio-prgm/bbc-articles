import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import random

# -------------------------------
# Set seeds for reproducibility
# -------------------------------
seed_val = 42
np.random.seed(seed_val)
random.seed(seed_val)
tf.random.set_seed(seed_val)

# Check for GPU availability
print("GPU Available:", tf.config.list_physical_devices('GPU'))

# -------------------------------
# Dataset Download and Loading
# -------------------------------
import kagglehub

def load_dataset():
    # Download the latest version of the dataset
    path = kagglehub.dataset_download("amunsentom/article-dataset-2")
    print("Path to dataset files:", path)

    # Look for CSV files in the downloaded folder
    csv_files = [f for f in os.listdir(path) if f.endswith('.csv')]
    if csv_files:
        dataset_path = os.path.join(path, csv_files[0])
        df = pd.read_csv(dataset_path)
        print("Dataset preview:")
        print(df.head())
        return df
    else:
        raise FileNotFoundError("No CSV files found in the downloaded folder.")

# -------------------------------
# Sentiment Labeling using VADER
# -------------------------------
import nltk
nltk.download('vader_lexicon')
from nltk.sentiment import SentimentIntensityAnalyzer

# Adjust thresholds to widen the neutral band
def vader_sentiment_label(text, pos_threshold=0.2, neg_threshold=-0.2):
    scores = sia.polarity_scores(text)
    compound = scores['compound']
    if compound >= pos_threshold:
        return 'positive'
    elif compound <= neg_threshold:
        return 'negative'
    else:
        return 'neutral'

sia = SentimentIntensityAnalyzer()

def label_sentiments(df):
    df['sentiment'] = df['text'].apply(lambda x: vader_sentiment_label(x))
    print("Sentiment distribution using VADER (adjusted thresholds):")
    print(df['sentiment'].value_counts())
    return df

# -------------------------------
# Preprocessing: Splitting and Tokenization
# -------------------------------
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

def preprocess_data(df, max_words=20000, maxlen=300):
    # Map sentiments to numerical labels
    sentiment_map = {'positive': 0, 'neutral': 1, 'negative': 2}
    X = df['text'].astype(str).values
    Y = np.array([sentiment_map[label] for label in df['sentiment']])

    # Stratified train-validation split
    X_train, X_val, Y_train, Y_val = train_test_split(
        X, Y, test_size=0.2, stratify=Y, random_state=seed_val
    )

    # Tokenization
    tokenizer = Tokenizer(num_words=max_words)
    tokenizer.fit_on_texts(X_train)
    X_train_seq = tokenizer.texts_to_sequences(X_train)
    X_val_seq   = tokenizer.texts_to_sequences(X_val)

    # Padding sequences
    X_train_pad = pad_sequences(X_train_seq, maxlen=maxlen, padding='post', truncating='post')
    X_val_pad   = pad_sequences(X_val_seq, maxlen=maxlen, padding='post', truncating='post')

    # One-hot encode labels
    Y_train_cat = to_categorical(Y_train, num_classes=3)
    Y_val_cat   = to_categorical(Y_val, num_classes=3)

    return tokenizer, X_train_pad, X_val_pad, Y_train_cat, Y_val_cat, Y_train

# -------------------------------
# Compute Class Weights
# -------------------------------
from sklearn.utils.class_weight import compute_class_weight

def get_class_weights(Y_train):
    classes = np.unique(Y_train)
    class_weights_array = compute_class_weight('balanced', classes=classes, y=Y_train)
    class_weights = {i: weight for i, weight in enumerate(class_weights_array)}
    print("Computed class weights:", class_weights)
    return class_weights

# -------------------------------
# Load Pre-trained GloVe Embeddings using Gensim
# -------------------------------
import gensim.downloader as api

def load_glove_embeddings(tokenizer, embedding_model, embedding_dim):
    """
    Build an embedding matrix from a pre-loaded GloVe model.
    """
    word_index = tokenizer.word_index
    embedding_matrix = np.zeros((len(word_index) + 1, embedding_dim))
    for word, i in word_index.items():
        if word in embedding_model:
            embedding_matrix[i] = embedding_model[word]
    return embedding_matrix, len(word_index) + 1

# -------------------------------
# Define the Improved LSTM Model
# -------------------------------
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

def build_model(vocab_size, embedding_dim, maxlen, embedding_matrix):
    model = Sequential([
        Embedding(input_dim=vocab_size,
                  output_dim=embedding_dim,
                  weights=[embedding_matrix],
                  input_length=maxlen,
                  trainable=False),
        Bidirectional(LSTM(units=64,
                           dropout=0.3,
                           recurrent_dropout=0.3,
                           return_sequences=True)),
        Dropout(0.3),
        Bidirectional(LSTM(units=32,
                           dropout=0.3,
                           recurrent_dropout=0.3)),
        Dropout(0.3),
        Dense(3, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    return model

# -------------------------------
# Plot Training History
# -------------------------------
def plot_history(history):
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss')

    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Accuracy')
    plt.show()

# -------------------------------
# Predict Sentiment for New Articles
# -------------------------------
def predict_sentiment(text, tokenizer, model, maxlen=300):
    seq = tokenizer.texts_to_sequences([text])
    pad_seq = pad_sequences(seq, maxlen=maxlen, padding='post', truncating='post')
    prediction = model.predict(pad_seq)
    sentiment = np.argmax(prediction)
    reverse_map = {0: 'positive', 1: 'neutral', 2: 'negative'}
    return reverse_map[sentiment]

# -------------------------------
# Main execution
# -------------------------------
def main():
    # Load and label dataset
    df = load_dataset()
    df = label_sentiments(df)

    # Preprocess data
    max_words = 20000
    maxlen = 300
    tokenizer, X_train_pad, X_val_pad, Y_train_cat, Y_val_cat, Y_train = preprocess_data(df, max_words, maxlen)

    # Compute class weights
    class_weights = get_class_weights(Y_train)

    # -------------------------------
    # Load pre-trained GloVe embeddings (50-dimensional) using Gensim
    # -------------------------------
    print("[Step 2] Loading pre-trained GloVe embeddings (50-dimensional)...")
    embedding_model = api.load('glove-wiki-gigaword-50')
    embedding_dim = 50
    print(f"[Step 2] Pre-trained GloVe embeddings (dim={embedding_dim}) loaded.")

    # Build embedding matrix
    embedding_matrix, vocab_size = load_glove_embeddings(tokenizer, embedding_model, embedding_dim)

    # Build the model
    model = build_model(vocab_size, embedding_dim, maxlen, embedding_matrix)

    # Callbacks: early stopping and learning rate reduction on plateau
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    lr_reduce = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)

    epochs = 50
    history = model.fit(
        X_train_pad, Y_train_cat,
        epochs=epochs,
        batch_size=32,
        validation_data=(X_val_pad, Y_val_cat),
        class_weight=class_weights,
        callbacks=[early_stop, lr_reduce]
    )

    # Plot training history
    plot_history(history)

    # Evaluate the model using a confusion matrix
    from sklearn.metrics import confusion_matrix
    y_pred = model.predict(X_val_pad)
    y_pred_classes = y_pred.argmax(axis=1)
    y_true = Y_val_cat.argmax(axis=1)

    cm = confusion_matrix(y_true, y_pred_classes)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Positive', 'Neutral', 'Negative'],
                yticklabels=['Positive', 'Neutral', 'Negative'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.show()

    # Test predictions on new texts
    test_texts = [
        # Positive examples
        "good excellent positive fortunate benefit win.",
        "Local communities celebrate after a successful campaign to protect a vital nature reserve from development.",
        "The government has announced new measures to combat the rising cost of living, including tax cuts for low-income families.",

        # Neutral examples
        "The company has announced its quarterly earnings, which showed steady growth despite market fluctuations.",
        "A new infrastructure project is set to begin next month, with expected completion in two years.",
        "The government is reviewing its policies on immigration in an effort to improve national security.",
        "Researchers are investigating the potential impact of a new drug on human health, with initial tests showing mixed results.",
        "The city council is planning to hold a public consultation regarding proposed changes to local zoning laws.",

        # Negative examples
        "The local hospital is facing severe budget cuts, leading to staff layoffs and reduced services for patients.",
        "A cyberattack has compromised sensitive personal information of thousands of users across the country.",
        "Protests erupted across the city following the controversial decision to raise fuel prices by 20%.",
        "The company’s recent product launch has faced significant backlash, with customers expressing dissatisfaction over quality issues.",
        "The economic slowdown is causing widespread job losses, with many small businesses struggling to stay afloat."
    ]

    print("\n--- Test Predictions ---\n")
    for i, text in enumerate(test_texts):
        sentiment = predict_sentiment(text, tokenizer, model, maxlen)
        print(f"Test {i+1}: {text}\nPredicted Sentiment: {sentiment}\n")

if __name__ == '__main__':
    main()
