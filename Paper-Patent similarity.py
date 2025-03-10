import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from tqdm import tqdm
import os
# Initialize BERT model
model = SentenceTransformer('all-MiniLM-L6-v2')
# If multiple GPUs are available, set the model to run in parallel on multiple GPUs
if torch.cuda.is_available():
    device = torch.device("cuda")
    model = model.to(device)
else:
    device = torch.device("cpu")
# Read data
patents_df = pd.read_csv('./DeepPatentAI.csv')
papers_df = pd.read_csv('./DeepDiveAI.csv')
# Clean keywords
papers_df['cleaned_keywords'] = papers_df['Keywords'].apply(lambda x: [word.strip() for word in eval(x) if word.strip() != ''])
patents_df['cleaned_keywords'] = patents_df['Keywords'].apply(lambda x: [word.strip() for word in eval(x) if word.strip() != ''])
# Convert keyword lists to text
papers_df['keywords_text'] = papers_df['cleaned_keywords'].apply(lambda x: ' '.join(x))
patents_df['keywords_text'] = patents_df['cleaned_keywords'].apply(lambda x: ' '.join(x))
# Convert text to model embeddings and save
def compute_and_save_embeddings(df, file_path, batch_size=512):
    if os.path.exists(file_path):
        print(f"Loading existing embeddings file: {file_path}")
        return np.load(file_path)
    
    print(f"Computing embeddings and saving to: {file_path}")
    embeddings = []
    for i in tqdm(range(0, len(df), batch_size)):
        batch = df['keywords_text'].iloc[i:i + batch_size].tolist()
        batch_embeddings = model.encode(batch, show_progress_bar=False)
        embeddings.append(batch_embeddings)
    
    all_embeddings = np.vstack(embeddings)
    np.save(file_path, all_embeddings)
    return all_embeddings
# Compute and save embeddings
paper_embeddings = compute_and_save_embeddings(papers_df, 'paper_embeddings.npy')
patent_embeddings = compute_and_save_embeddings(patents_df, 'patent_embeddings.npy')
# Calculate similarity and write directly to file
def calculate_and_save_similarity(paper_embeddings, patent_embeddings, 
                                 paper_ids, patent_ids,
                                 output_file, 
                                 paper_batch_size=50000, 
                                 patent_batch_size=30000):
    
    # Create or clear output file
    with open(output_file, 'w') as f:
        # Write header row
        header = "paper_id,patent_id,similarity\n"
        f.write(header)
    
    # Process paper embeddings in batches
    for i in tqdm(range(0, len(paper_embeddings), paper_batch_size)):
        paper_batch = paper_embeddings[i:i + paper_batch_size]
        paper_ids_batch = paper_ids[i:i + paper_batch_size]
        
        # Process patent embeddings in batches
        for j in range(0, len(patent_embeddings), patent_batch_size):
            patent_batch = patent_embeddings[j:j + patent_batch_size]
            patent_ids_batch = patent_ids[j:j + patent_batch_size]
            
            # Calculate similarity between these two batches
            batch_similarities = cosine_similarity(paper_batch, patent_batch)
            
            # Convert results to CSV format and write to file
            with open(output_file, 'a') as f:
                for p_idx, paper_id in enumerate(paper_ids_batch):
                    for pt_idx, patent_id in enumerate(patent_ids_batch):
                        sim = batch_similarities[p_idx, pt_idx]
                        # Only save similarities above threshold to save space
                        if sim > 0.6:  # Example threshold, can be adjusted as needed
                            # Round similarity to 3 decimal places
                            sim_rounded = round(sim, 3)
                            f.write(f"{paper_id},{patent_id},{sim_rounded}\n")
# Calculate and save similarity matrix
calculate_and_save_similarity(
    paper_embeddings, 
    patent_embeddings,
    papers_df['ID'].tolist(),
    patents_df['ID'].tolist(),
    'similarity_results.csv'
)
print("Calculation complete, results saved to similarity_results.csv")