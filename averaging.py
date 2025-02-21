import time
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
import string
import numpy as np
import gensim.downloader as api
from sklearn.metrics.pairwise import cosine_similarity

# Download the tokenizer data if needed
print("Downloading NLTK data (if not already present)...")
nltk.download('punkt_tab')

# -------------------------------
# Step 1: Data Ingestion & Preprocessing
# -------------------------------
print("\n[Step 1] Starting data ingestion and preprocessing...")
start = time.time()
df = pd.read_csv('bbc-text-1.csv')  # CSV with 'category' and 'text'
articles = df['text'].tolist()
print(f"[Step 1] Loaded {len(articles)} articles.")

# Clean and tokenize each article
translator = str.maketrans('', '', string.punctuation)
tokenized_articles = []
for i, article in enumerate(articles):
    if (i+1) % 50 == 0 or i == 0 or i == len(articles)-1:
        print(f"[Step 1] Processing article {i+1}/{len(articles)}")
    # Lowercase, remove punctuation, and tokenize
    clean_text = article.lower().translate(translator)
    tokens = word_tokenize(clean_text)
    tokenized_articles.append(tokens)
    
preproc_time = time.time() - start
print(f"[Step 1] Data loading and preprocessing completed in {preproc_time:.2f} sec.")

# -------------------------------
# Step 2: Load Pre-trained Embeddings & Compute Article Embeddings
# -------------------------------
print("\n[Step 2] Loading pre-trained embeddings...")
start = time.time()
# Load a pre-trained embedding model (using a small model for speed)
model = api.load('glove-wiki-gigaword-50')  # 50-dimensional embeddings
load_model_time = time.time() - start
print(f"[Step 2] Pre-trained embeddings loaded in {load_model_time:.2f} sec.")

def average_embedding(tokens, model):
    # Compute the average embedding for the list of tokens
    vectors = [model[word] for word in tokens if word in model.key_to_index]
    return np.mean(vectors, axis=0) if vectors else np.zeros(model.vector_size)

print("[Step 2] Computing article embeddings (this may take a while)...")
start = time.time()
article_embeddings = []
for i, tokens in enumerate(tokenized_articles):
    if (i+1) % 50 == 0 or i == 0 or i == len(tokenized_articles)-1:
        print(f"[Step 2] Computing embedding for article {i+1}/{len(tokenized_articles)}")
    emb = average_embedding(tokens, model)
    article_embeddings.append(emb)
article_embeddings = np.array(article_embeddings)
embedding_time = time.time() - start
print(f"[Step 2] Article embedding (averaging) computed in {embedding_time:.2f} sec.")

# -------------------------------
# Step 3: Similarity Search (Recommendation)
# -------------------------------
print("\n[Step 3] Computing similarity matrix among articles...")
start = time.time()
similarity_matrix = cosine_similarity(article_embeddings)
sim_time = time.time() - start
print(f"[Step 3] Similarity computation completed in {sim_time:.2f} sec.")

# Example: Get top-5 similar articles for article index 0 (excluding itself)
article_idx = 0
sim_scores = similarity_matrix[article_idx].copy()
sim_scores[article_idx] = -1  # exclude self from recommendations
top5_idx = np.argsort(sim_scores)[-5:][::-1]
print(f"[Step 3] Top 5 similar articles for article {article_idx}: {top5_idx}\n" + 
      "\n".join([articles[article_id] for article_id in top5_idx]))

print("\n[All Steps Completed]")

# [ 228 1408  912 1893 1043]