import time
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
import string
import numpy as np
import gensim.downloader as api
from sklearn.metrics.pairwise import cosine_similarity

# -------------------------------
# Step 1: Data Ingestion & Preprocessing
# -------------------------------
print("Downloading NLTK data (if not already present)...")
nltk.download("punkt")

print("\n[Step 1] Loading and preprocessing data...")
start = time.time()
df = pd.read_csv("bbc-text-1.csv")  # CSV with 'category' and 'text'
articles = df["text"].tolist()
print(f"[Step 1] Loaded {len(articles)} articles.")

translator = str.maketrans("", "", string.punctuation)
tokenized_articles = []
for i, article in enumerate(articles):
    if (i + 1) % 50 == 0 or i == 0 or i == len(articles) - 1:
        print(f"[Step 1] Processing article {i+1}/{len(articles)}")
    clean_text = article.lower().translate(translator)
    tokens = word_tokenize(clean_text)
    tokenized_articles.append(tokens)

preproc_time = time.time() - start
print(f"[Step 1] Completed in {preproc_time:.2f} sec.")

# -------------------------------
# Step 2: Load Pre-trained Embeddings & Compute Article Embeddings
# -------------------------------
print("\n[Step 2] Loading pre-trained GloVe embeddings...")
start = time.time()
model = api.load("glove-wiki-gigaword-50")  # 50-dimensional embeddings
load_model_time = time.time() - start
print(f"[Step 2] Embeddings loaded in {load_model_time:.2f} sec.")


def average_embedding(tokens, model):
    vectors = [model[word] for word in tokens if word in model.key_to_index]
    return np.mean(vectors, axis=0) if vectors else np.zeros(model.vector_size)


print("[Step 2] Computing averaged embeddings for articles...")
start = time.time()
article_embeddings = []
for i, tokens in enumerate(tokenized_articles):
    if (i + 1) % 50 == 0 or i == 0 or i == len(tokenized_articles) - 1:
        print(f"[Step 2] Processing article {i+1}/{len(tokenized_articles)}")
    emb = average_embedding(tokens, model)
    article_embeddings.append(emb)
article_embeddings = np.array(article_embeddings)
embedding_time = time.time() - start
print(f"[Step 2] Computed embeddings in {embedding_time:.2f} sec.")

# -------------------------------
# Step 3: Compute Similarity Matrix
# -------------------------------
print("\n[Step 3] Computing cosine similarity matrix...")
start = time.time()
similarity_matrix = cosine_similarity(article_embeddings)
sim_time = time.time() - start
print(f"[Step 3] Similarity matrix computed in {sim_time:.2f} sec.")

# -------------------------------
# Step 4: Use Fixed Test Indices and Save Results to a Text File
# -------------------------------
print(
    "\n[Step 4] Retrieving top-3 recommendations for fixed test articles and saving to 'test_results.txt'...\n"
)

# Fixed test indices
test_indices = [1068, 247, 1778, 1782, 1332]
top_n = 3  # number of recommendations per article

# Open a text file to save the results
with open("ave_test_results.txt", "w", encoding="utf-8") as f:
    for i, idx in enumerate(test_indices, 1):
        # Copy similarity scores and explicitly set the score for the article itself to -1
        sim_scores = similarity_matrix[idx].copy()
        sim_scores[idx] = -1

        # Get indices for the top-3 similar articles (sorted in descending order by similarity)
        top_indices = np.argsort(sim_scores)[-top_n:][::-1]

        # Build the output text
        output_text = f"--- Test {i} ---\n"
        output_text += f"Test Article Index: {idx}\n"
        output_text += f"Test Article:\n{articles[idx]}\n\n"
        output_text += "Top 3 Similar Articles:\n"
        for rank, sim_idx in enumerate(top_indices, 1):
            output_text += f"{rank}. Article Index {sim_idx}:\n{articles[sim_idx]}\n\n"
        output_text += "\n" + ("-" * 80) + "\n\n"

        # Write to the file and also print to console
        f.write(output_text)
        print(output_text)

print("Results saved to 'test_results.txt'.")
