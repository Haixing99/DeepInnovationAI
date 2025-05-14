import torch
from sentence_transformers import SentenceTransformer
import pandas as pd
from torch.nn.functional import cosine_similarity
from tqdm import tqdm
import os

# Configurable parameters
PAPERS_PATH = 'DeepDiveAI.csv'       # Paper dataset path
PATENTS_PATH = 'DeepPatentAI.csv'       # Patent dataset path
OUTPUT_PATH = 'DeepCosineAI.csv'   # Output file path
MODEL_NAME = 'all-MiniLM-L6-v2'         # Pre-trained model
BATCH_SIZE = 512                        # Batch size for embedding calculation
SIMILARITY_BATCH = 10000                # Batch size for similarity calculation

# Set GPU device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device used: {device}")

# Load model
print(f"Loading model: {MODEL_NAME}")
model = SentenceTransformer(MODEL_NAME)

# Load data
print("Loading data...")
papers_df = pd.read_csv(PAPERS_PATH)
patents_df = pd.read_csv(PATENTS_PATH)

# Preprocess keywords (assuming Keywords is a string list)
papers_df['keywords_text'] = papers_df['Keywords'].apply(lambda x: ' '.join(eval(x) if isinstance(x, str) else x))
patents_df['keywords_text'] = patents_df['Keywords'].apply(lambda x: ' '.join(eval(x) if isinstance(x, str) else x))

# Function to compute embeddings
def compute_embeddings(texts, model, batch_size=BATCH_SIZE):
    embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Computing embeddings"):
        batch = texts[i:i + batch_size]
        batch_embeddings = model.encode(batch, convert_to_tensor=True)
        embeddings.append(batch_embeddings.cpu())  # Store on CPU
    return torch.cat(embeddings, dim=0)

# Compute and save embeddings
if not os.path.exists('paper_embeddings.pt') or not os.path.exists('patent_embeddings.pt'):
    print("Computing patent embeddings...")
    patent_embeddings = compute_embeddings(patents_df['keywords_text'].tolist(), model)
    torch.save(patent_embeddings, 'patent_embeddings.pt')
    print("Computing paper embeddings...")
    paper_embeddings = compute_embeddings(papers_df['keywords_text'].tolist(), model)
    torch.save(paper_embeddings, 'paper_embeddings.pt')

# Load embeddings to CPU, and set weights_only=True to avoid warnings
print("Loading embeddings...")
patent_embeddings = torch.load('patent_embeddings.pt', map_location='cpu', weights_only=True)
paper_embeddings = torch.load('paper_embeddings.pt', map_location='cpu', weights_only=True)

# Create output file and write header, adding paper and patent year columns
print("Initializing output file...")
with open(OUTPUT_PATH, 'w') as f:
    f.write('Paper_ID,Patent_ID,Similarity,Paper_Year,Patent_Year\n')

# Calculate similarity and save immediately, ensuring patent year is later than paper year
print("Calculating similarity and saving immediately...")
for i in tqdm(range(len(papers_df)), desc="Processing papers"):
    paper_embedding = paper_embeddings[i].unsqueeze(0).to(device)  # Single paper embedding to GPU
    paper_year = papers_df.iloc[i]['Year']  # Paper year
    max_sim = -1
    max_patent_id = None
    max_patent_year = None  # Save corresponding patent year
    
    # Filter patent indices where year is later than paper year
    valid_patent_indices = patents_df.index[patents_df['Year'] >= paper_year].tolist()
    if len(valid_patent_indices) == 0:
        continue  # No patents meeting the year condition, skip this paper
    
    # Process qualifying patent embeddings in batches
    for j in range(0, len(valid_patent_indices), SIMILARITY_BATCH):
        batch_indices = valid_patent_indices[j:j + SIMILARITY_BATCH]
        batch_embeddings = patent_embeddings[batch_indices].to(device)
        sim = cosine_similarity(paper_embedding, batch_embeddings)[0]
        batch_max_sim, batch_max_idx = torch.max(sim, dim=0)
        if batch_max_sim > max_sim:
            max_sim = batch_max_sim
            max_patent_id = patents_df.iloc[batch_indices[batch_max_idx.item()]]['ID']
            max_patent_year = patents_df.iloc[batch_indices[batch_max_idx.item()]]['Year']
        batch_embeddings = batch_embeddings.cpu()
        torch.cuda.empty_cache()
    
    # Save individual result to CSV immediately, including paper and patent years
    if max_patent_id is not None:
        with open(OUTPUT_PATH, 'a') as f:
            f.write(f"{papers_df.iloc[i]['ID']},{max_patent_id},{max_sim.item()},{paper_year},{max_patent_year}\n")

print(f"Calculation complete! Results saved to {OUTPUT_PATH}")
