import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import os

"""
Paper-Patent Similarity Calculator for DeepInnovationAI Dataset
This script calculates semantic similarity between academic papers and patents.
Researchers can modify the paths and parameters below according to their needs.
"""

# Configurable parameters - researchers can modify as needed
PAPERS_PATH = 'DeepDiveAI.csv'  # Path to papers data
PATENTS_PATH = 'DeepPatentAI.csv'  # Path to patents data
OUTPUT_PATH = 'similarity_matrix.csv'  # Output file path
PAPERS_LIMIT = 10000  # Number of papers to process, -1 for all
PATENTS_LIMIT = 10000  # Number of patents to process, -1 for all
MODEL_NAME = 'all-MiniLM-L6-v2'  # Pretrained model name
BATCH_SIZE = 512  # Encoding batch size
SIMILARITY_BATCH = 1000  # Similarity calculation batch size
SIMILARITY_THRESHOLD = 0.6  # High similarity threshold for pair extraction

# Initialize BERT model
print(f"Loading model: {MODEL_NAME}")
model = SentenceTransformer(MODEL_NAME)

# Set device (GPU or CPU)
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using GPU for computation")
else:
    device = torch.device("cpu")
    print("Using CPU for computation")
model = model.to(device)

# Load data
print(f"Reading patent data: {PATENTS_PATH}")
patents_df = pd.read_csv(PATENTS_PATH)
if PATENTS_LIMIT > 0:
    patents_df = patents_df.iloc[0:PATENTS_LIMIT, :]

print(f"Reading paper data: {PAPERS_PATH}")
papers_df = pd.read_csv(PAPERS_PATH)
if PAPERS_LIMIT > 0:
    papers_df = papers_df.iloc[0:PAPERS_LIMIT, :]

print(f"Processing {len(papers_df)} papers and {len(patents_df)} patents")

# Clean keywords
print("Processing keywords...")
try:
    papers_df['cleaned_keywords'] = papers_df['Keywords'].apply(lambda x: [word.strip() for word in eval(x) if word.strip() != ''])
    patents_df['cleaned_keywords'] = patents_df['Keywords'].apply(lambda x: [word.strip() for word in eval(x) if word.strip() != ''])
except Exception as e:
    print(f"Error processing keywords: {e}, trying alternative method")
    # Fallback processing method
    papers_df['cleaned_keywords'] = papers_df['Keywords'].apply(
        lambda x: str(x).replace('[', '').replace(']', '').replace("'", "").split(','))
    patents_df['cleaned_keywords'] = patents_df['Keywords'].apply(
        lambda x: str(x).replace('[', '').replace(']', '').replace("'", "").split(','))

# Convert keyword lists to text
papers_df['keywords_text'] = papers_df['cleaned_keywords'].apply(lambda x: ' '.join([w.strip() for w in x if w.strip()]))
patents_df['keywords_text'] = patents_df['cleaned_keywords'].apply(lambda x: ' '.join([w.strip() for w in x if w.strip()]))

# Convert texts to model embeddings
def batch_encode(texts, batch_size=BATCH_SIZE):
    """Batch computation of text embeddings"""
    embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Computing embeddings"):
        batch = texts[i:i + batch_size]
        batch_embeddings = model.encode(batch, show_progress_bar=False, convert_to_tensor=True)
        embeddings.append(batch_embeddings)
    return torch.cat(embeddings, dim=0)

# Calculate embeddings
print("Computing paper embeddings...")
paper_embeddings = batch_encode(papers_df['keywords_text'].tolist())
print("Computing patent embeddings...")
patent_embeddings = batch_encode(patents_df['keywords_text'].tolist())

# Calculate cosine similarity
def calculate_similarity(paper_embeddings, patent_embeddings, batch_size=SIMILARITY_BATCH):
    """Batch calculation of paper-patent similarity"""
    similarity_matrix = []
    for i in tqdm(range(0, len(paper_embeddings), batch_size), desc="Calculating similarity"):
        paper_batch = paper_embeddings[i:i + batch_size]
        batch_sim = cosine_similarity(paper_batch.cpu(), patent_embeddings.cpu())
        similarity_matrix.append(torch.tensor(batch_sim))
    return torch.cat(similarity_matrix, axis=0)

print(f"Calculating similarity matrix ({len(paper_embeddings)} × {len(patent_embeddings)})...")
similarity_matrix = calculate_similarity(paper_embeddings, patent_embeddings)

# Convert results to DataFrame and save
print(f"Saving complete similarity matrix to {OUTPUT_PATH}")
similarity_df = pd.DataFrame(similarity_matrix.numpy(), columns=patents_df['ID'], index=papers_df['ID'])
similarity_df.to_csv(OUTPUT_PATH)

# Extract and save high similarity pairs (optional)
high_sim_pairs = []
print(f"Extracting paper-patent pairs with similarity >= {SIMILARITY_THRESHOLD}...")
for i in tqdm(range(similarity_matrix.shape[0]), desc="Processing high similarity pairs"):
    for j in range(similarity_matrix.shape[1]):
        if similarity_matrix[i, j] >= SIMILARITY_THRESHOLD:
            high_sim_pairs.append({
                'Paper_ID': papers_df.iloc[i]['ID'],
                'Patent_ID': patents_df.iloc[j]['ID'],
                'Similarity': similarity_matrix[i, j].item()
            })

if high_sim_pairs:
    high_sim_df = pd.DataFrame(high_sim_pairs)
    high_sim_df.sort_values('Similarity', ascending=False, inplace=True)
    high_sim_path = 'high_similarity_pairs.csv'
    high_sim_df.to_csv(high_sim_path, index=False)
    print(f"Found {len(high_sim_pairs)} high similarity paper-patent pairs, saved to {high_sim_path}")

print("Calculation completed!")
