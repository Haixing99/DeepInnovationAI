import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import os

"""
Paper-Patent Similarity Calculator for DeepInnovationAI Dataset
This script calculates semantic similarity between academic papers and patents.
Output is a table with Paper ID, Patent ID, and similarity score.
"""

# Configurable parameters - researchers can modify as needed
PAPERS_PATH = '../DeepDiveAI.csv'  # Path to papers data
PATENTS_PATH = '../DeepPatentAI.csv'  # Path to patents data
OUTPUT_PATH = '../DeepCosineAI.csv'  # Output file path
PAPERS_LIMIT = 10000  # Number of papers to process, -1 for all
PATENTS_LIMIT = 10000  # Number of patents to process, -1 for all
MODEL_NAME = 'all-MiniLM-L6-v2'  # Pretrained model name
BATCH_SIZE = 512  # Encoding batch size
SIMILARITY_BATCH = 1000  # Similarity calculation batch size
SIMILARITY_THRESHOLD = 0.0  # Minimum similarity threshold (0 = save all pairs)

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

# Calculate cosine similarity and directly save as pairs to save memory
def calculate_similarity_pairs(paper_embeddings, patent_embeddings, papers_df, patents_df, 
                             batch_size=SIMILARITY_BATCH, threshold=SIMILARITY_THRESHOLD, 
                             output_path=OUTPUT_PATH):
    """
    Calculate similarity and directly output as pairs to save memory
    """
    # Create output file and write header
    with open(output_path, 'w') as f:
        f.write("Paper_ID,Patent_ID,Similarity\n")
    
    total_pairs = 0
    
    # Process in batches to save memory
    for i in tqdm(range(0, len(paper_embeddings), batch_size), desc="Calculating similarity pairs"):
        paper_batch = paper_embeddings[i:i + batch_size]
        paper_ids = papers_df['ID'].iloc[i:i + batch_size].tolist()
        
        # Calculate similarity for this batch against all patents
        batch_sim = cosine_similarity(paper_batch.cpu(), patent_embeddings.cpu())
        
        # Convert to pairs and save directly to file
        pairs_batch = []
        for j in range(batch_sim.shape[0]):
            for k in range(batch_sim.shape[1]):
                similarity = batch_sim[j, k]
                if similarity >= threshold:
                    pairs_batch.append(f"{paper_ids[j]},{patents_df['ID'].iloc[k]},{similarity:.6f}\n")
        
        # Write batch to file
        with open(output_path, 'a') as f:
            f.writelines(pairs_batch)
        
        total_pairs += len(pairs_batch)
        
    print(f"Total {total_pairs} similarity pairs saved to {output_path}")
    return total_pairs

print(f"Calculating and saving similarity pairs...")
total_pairs = calculate_similarity_pairs(paper_embeddings, patent_embeddings, 
                                      papers_df, patents_df, 
                                      threshold=SIMILARITY_THRESHOLD)

print(f"Calculation completed! {total_pairs} pairs saved to {OUTPUT_PATH}")

# If you need to load the results for analysis later:
# similarity_pairs = pd.read_csv(OUTPUT_PATH)
# Top similarity pairs:
# top_pairs = similarity_pairs.sort_values('Similarity', ascending=False).head(100)
