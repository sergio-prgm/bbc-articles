import time
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
import string
import numpy as np
import gensim.downloader as api
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from sklearn.manifold import TSNE

print("Downloading required NLTK data (punkt) if not already present...")
nltk.download('punkt')

# -------------------------------
# Step 1: Data Ingestion & Preprocessing
# -------------------------------
print("\n[Step 1] Starting data ingestion and preprocessing...")
start = time.time()
df = pd.read_csv('bbc-text-1.csv')  # CSV with 'category' and 'text'
articles = df['text'].tolist()
categories = df['category'].tolist()
print(f"[Step 1] Loaded {len(articles)} articles with categories.")

translator = str.maketrans('', '', string.punctuation)
tokenized_articles = []
for i, article in enumerate(articles):
    if (i + 1) % 50 == 0 or i == 0 or i == len(articles) - 1:
        print(f"[Step 1] Processing article {i + 1}/{len(articles)}")
    clean_text = article.lower().translate(translator)
    tokens = word_tokenize(clean_text)
    tokenized_articles.append(tokens)
preproc_time = time.time() - start
print(f"[Step 1] Data loading and preprocessing completed in: {preproc_time:.2f} sec")

# Convert tokens back into strings for the tokenizer.
texts = [" ".join(tokens) for tokens in tokenized_articles]

# -------------------------------
# Step 2: Prepare Sequences & Load Pre-trained Embeddings
# -------------------------------
print("\n[Step 2] Preparing sequences and loading pre-trained embeddings...")
start = time.time()
max_length = 200  # Maximum sequence length

# Create a tokenizer and fit on the texts
tokenizer = Tokenizer()
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
padded_sequences = pad_sequences(sequences, maxlen=max_length, padding='post')
vocab_size = len(tokenizer.word_index) + 1
print(f"[Step 2] Vocabulary size: {vocab_size}, Sequence shape: {padded_sequences.shape}")

# Load pre-trained GloVe embeddings (50-dimensional)
embedding_model = api.load('glove-wiki-gigaword-50')
embedding_dim = 50
print(f"[Step 2] Pre-trained GloVe embeddings (dim={embedding_dim}) loaded.")

# Create an embedding matrix for our vocabulary
embedding_matrix = np.zeros((vocab_size, embedding_dim))
for word, i in tokenizer.word_index.items():
    if word in embedding_model.key_to_index:
        embedding_matrix[i] = embedding_model[word]
    else:
        # Initialize missing words with random values
        embedding_matrix[i] = np.random.normal(scale=0.6, size=(embedding_dim,))
embed_time = time.time() - start
print(f"[Step 2] Sequences prepared and embedding matrix built in: {embed_time:.2f} sec")

# -------------------------------
# Step 3: Build and Train an Improved LSTM Model
# -------------------------------
print("\n[Step 3] Building and training an improved bidirectional LSTM model...")

# Encode labels (categories) to integers
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(categories)
num_classes = len(label_encoder.classes_)
print(f"[Step 3] Number of classes: {num_classes}")

# Split data into train and validation sets
X_train, X_val, y_train, y_val = train_test_split(
    padded_sequences, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

# Build the model
model_lstm = Sequential([
    Embedding(input_dim=vocab_size,
              output_dim=embedding_dim,
              weights=[embedding_matrix],
              trainable=True),  # Allow fine-tuning of embeddings
    Bidirectional(LSTM(64, dropout=0.2, recurrent_dropout=0.2, return_sequences=True), name='bilstm_layer_1'),
    Bidirectional(LSTM(64, dropout=0.2, recurrent_dropout=0.2), name='bilstm_layer_2'),
    Dropout(0.2),
    Dense(num_classes, activation='softmax')
])
model_lstm.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
print("[Step 3] LSTM model built and compiled.")

# Define callbacks for early stopping and model checkpointing
callbacks = [
    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
    # ModelCheckpoint('best_model.keras', monitor='val_loss', save_best_only=True)
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)
]

start = time.time()
print("[Step 3] Starting training...")
history = model_lstm.fit(X_train, y_train, 
                         epochs=50, 
                         batch_size=32, 
                         validation_data=(X_val, y_val),
                         callbacks=callbacks,
                         verbose=1)
lstm_train_time = time.time() - start
print(f"[Step 3] LSTM training completed in: {lstm_train_time:.2f} sec")

# Plot Loss and Accuracy
# plt.figure(figsize=(14, 5))

# plt.subplot(1, 2, 1)
# plt.plot(history.history['loss'], label='Training Loss', marker='o')
# plt.plot(history.history['val_loss'], label='Validation Loss', marker='o')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.legend()
# plt.title('Training vs. Validation Loss')

# plt.subplot(1, 2, 2)
# plt.plot(history.history['accuracy'], label='Training Accuracy', marker='o')
# plt.plot(history.history['val_accuracy'], label='Validation Accuracy', marker='o')
# plt.xlabel('Epochs')
# plt.ylabel('Accuracy')
# plt.legend()
# plt.title('Training vs. Validation Accuracy')
# plt.tight_layout()
# plt.show()

#
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
# Step 4: Evaluate the Model with a Confusion Matrix
# -------------------------------
print("\n[Step 4] Evaluating model performance on the validation set...")
y_val_pred = model_lstm.predict(X_val)
y_val_pred_labels = np.argmax(y_val_pred, axis=1)

cm = confusion_matrix(y_val, y_val_pred_labels)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap='Blues',
            xticklabels=label_encoder.classes_,
            yticklabels=label_encoder.classes_)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

print("\nClassification Report:")
print(classification_report(y_val, y_val_pred_labels, target_names=label_encoder.classes_))

# -------------------------------
# Step 5: Extract Refined Article Embeddings & Visualization
# -------------------------------
print("\n[Step 5] Extracting refined article embeddings from the second BiLSTM layer...")
start = time.time()
# Create a model to extract outputs from the second bidirectional LSTM layer
embedding_extractor = Model(inputs=model_lstm.inputs, 
                            outputs=model_lstm.get_layer('bilstm_layer_2').output)
refined_embeddings = embedding_extractor.predict(padded_sequences)
extract_time = time.time() - start
print(f"[Step 5] Refined article embeddings extracted in: {extract_time:.2f} sec")

# t-SNE visualization of embeddings colored by category
print("\n[Step 5] Generating t-SNE visualization of article embeddings...")
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
embeddings_2d = tsne.fit_transform(refined_embeddings)
plt.figure(figsize=(10, 8))
scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                      c=y_encoded, cmap='viridis', alpha=0.7)
# plt.legend(handles=scatter.legend_elements()[0], labels=label_encoder.classes_, title="Categories")
handles, _ = scatter.legend_elements()
handles = list(handles)  # Ensure it's a list
labels = list(label_encoder.classes_)  # Convert label_encoder.classes_ to a list
plt.legend(handles=handles, labels=labels, title="Categories")

plt.title('t-SNE Visualization of Article Embeddings')
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.show()

# -------------------------------
# Step 6: Similarity Search & Heatmap
# -------------------------------
print("\n[Step 6] Computing similarity matrix among refined embeddings...")
start = time.time()
similarity_matrix = cosine_similarity(refined_embeddings)
sim_time = time.time() - start
print(f"[Step 6] Similarity computation completed in: {sim_time:.2f} sec")

# article_idx = 0
# sim_scores = similarity_matrix[article_idx].copy()
# sim_scores[article_idx] = -1  # Exclude self from similarity search
# top5_idx = np.argsort(sim_scores)[-5:][::-1]
# print(f"[Step 6] Top 5 similar articles for article {article_idx}:")
# for idx in top5_idx:
#     print(f"Article {idx}: {articles[idx]}")
import random

# Determine the maximum index to sample from
max_index = min(2000, len(articles))

# Choose 2 random indices from 0 to max_index
random_indices = random.sample(range(max_index), 2)

for article_idx in random_indices:
    sim_scores = similarity_matrix[article_idx].copy()
    sim_scores[article_idx] = -1  # Exclude self from similarity search
    top5_idx = np.argsort(sim_scores)[-3:][::-1]
    print(f"\n[Step 6] Top 5 similar articles for article {article_idx} {articles[article_idx]}:")
    for idx in top5_idx:
        print(f"Article {idx}: {articles[idx]}")

# Plot a heatmap for a subset of articles (first 50) to visualize similarity
subset_size = 50
subset_similarity = cosine_similarity(refined_embeddings[:subset_size])
plt.figure(figsize=(10, 8))
sns.heatmap(subset_similarity, cmap='coolwarm')
plt.title('Cosine Similarity Heatmap for Subset of Articles')
plt.xlabel('Article Index')
plt.ylabel('Article Index')
plt.show()
