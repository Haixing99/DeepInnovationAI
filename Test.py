import torch
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding
)
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
from tqdm import tqdm
import os
import random
import numpy as np
from torch.cuda.amp import autocast

# Set random seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed()

# Check CUDA devices
def check_cuda_devices():
    """Check and print information about available CUDA devices"""
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        print(f"System has {num_gpus} GPU(s) available.")
        for i in range(num_gpus):
            print(f"GPU {i} model: {torch.cuda.get_device_name(i)}")
    else:
        print("No GPUs available in the current system.")

check_cuda_devices()

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. Data preprocessing function
def prepare_data(test_data):
    """
    Preprocess test data, return text and label lists.
    Assumes test_data contains 'title', 'abs', 'label' fields.
    Combines title and abs into a single input text field.
    
    Args:
        test_data (DataFrame): Test data containing title, abstract, and labels
        
    Returns:
        tuple: (test_texts, test_labels)
    """
    # Fill NA values with empty strings, then join title and abs with a space
    test_data['text'] = test_data[['title', 'abs']].fillna('').agg(' '.join, axis=1)
    test_texts = test_data['text'].tolist()
    test_labels = [int(label) for label in test_data['label'].tolist()]
    return test_texts, test_labels

# 2. Custom dataset class
class PatentDataset(Dataset):
    """
    Custom dataset class for sequence classification tasks.
    
    Args:
        texts (List[str]): List of text sequences
        labels (List[int]): List of corresponding labels
        tokenizer (PreTrainedTokenizer): Tokenizer for text tokenization
        max_length (int): Maximum text length (optional)
    """
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __getitem__(self, idx):
        """
        Get a single tokenized item with its label.
        
        Args:
            idx (int): Index of the item
            
        Returns:
            dict: Dictionary containing input_ids, attention_mask, and label
        """
        # Encode a single text, truncate only, let DataCollator handle dynamic padding
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        # Squeeze to remove dimensions (batch_size=1)
        item = {key: val.squeeze(0) for key, val in encoding.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.labels)
        
def evaluate_model(model, test_loader, device, mixed_precision=False):
    """
    Evaluate model on test set and calculate Precision, Recall, F1-score, and Accuracy.
    
    Args:
        model: The model to evaluate
        test_loader: DataLoader for test data
        device: Device to evaluate on
        mixed_precision (bool): Whether to use mixed precision for inference
        
    Returns:
        tuple: (metrics, true_labels, predictions)
    """
    model.eval()
    true_labels = []
    predictions = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            if mixed_precision:
                with autocast():
                    outputs = model(input_ids, attention_mask=attention_mask)
            else:
                outputs = model(input_ids, attention_mask=attention_mask)
                
            logits = outputs.logits
            batch_preds = torch.argmax(logits, dim=1).cpu().numpy()
            predictions.extend(batch_preds)
            true_labels.extend(labels.cpu().numpy())

    # Calculate classification report
    report = classification_report(true_labels, predictions, output_dict=True, zero_division=0)
    
    # Average metrics across classes (for binary classification - classes 0 and 1)
    metrics = {
        'precision': (report['0']['precision'] + report['1']['precision']) / 2,
        'recall': (report['0']['recall'] + report['1']['recall']) / 2,
        'f1-score': (report['0']['f1-score'] + report['1']['f1-score']) / 2,
        'accuracy': accuracy_score(true_labels, predictions),
    }

    # Print detailed classification report and summary metrics
    print("Detailed Classification Report:")
    print(classification_report(true_labels, predictions, zero_division=0, digits=3))
    print(f"Accuracy: {metrics['accuracy']:.4f}")

    return metrics, true_labels, predictions

if __name__ == "__main__":
    # Test set paths
    test_ai_path = './data/testing/AI_testing.csv'
    test_non_ai_path = './data/testing/NonAI_testing.csv'

    # Create results directory if it doesn't exist
    os.makedirs('./results', exist_ok=True)
    
    # Random sample subset of data for testing (adjust sample sizes as needed)
    test_ai = pd.read_csv(test_ai_path).sample(n=20000, random_state=42)
    test_non_ai = pd.read_csv(test_non_ai_path).sample(n=20000, random_state=42)
    test_data = pd.concat([test_ai, test_non_ai], ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Print test set class distribution
    print("Test set class distribution:")
    print(test_data['label'].value_counts())
    
    # Preprocess data, get texts and labels
    test_texts, test_labels = prepare_data(test_data)
    
    # Path to the trained best model
    model_path = "./models/BERT/BestBERT"
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    
    # If multiple GPUs are available, use DataParallel for parallel prediction
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs for prediction.")
        model = torch.nn.DataParallel(model)
    
    # Move model to specified device
    model.to(device)

    # Create test dataset instance
    test_dataset = PatentDataset(test_texts, test_labels, tokenizer)
    
    # Create data collator for dynamic padding
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    # Create DataLoader with appropriate batch size, disable shuffling
    test_loader = DataLoader(
        test_dataset, 
        batch_size=32,  # Adjust based on GPU count and memory
        shuffle=False, 
        collate_fn=data_collator,
        num_workers=4,  # Use multiple workers for data loading
        pin_memory=True  # Pin memory for faster data transfer to GPU
    )
    
    # Perform testing evaluation, calculate metrics, and get true and predicted labels
    metrics, true_labels, predictions = evaluate_model(
        model, 
        test_loader, 
        device,
        mixed_precision=True  # Enable mixed precision for faster inference
    )
    
    # 3. Save prediction results
    """
    Save true labels and predicted labels to a CSV file for subsequent analysis.
    Save path is in the results directory, filename is test_predictions.csv.
    """
    # Define complete path for saving predictions
    preds_save_path = './results/test_predictions.csv'
    
    # Create DataFrame containing true labels and predictions
    preds_df = pd.DataFrame({
        'true_labels': true_labels,
        'predictions': predictions
    })
    
    # Save to CSV file, index starts from 0
    preds_df.to_csv(preds_save_path, index=False, encoding='utf-8-sig')
    print(f"Prediction results saved to: {preds_save_path}")
    
    # Print summary metrics
    print("\nSummary of Test Results:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")