import torch
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AdamW,
    get_scheduler,
    DataCollatorWithPadding
)
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
from tqdm import tqdm
import os
import numpy as np
import random
from torch.cuda.amp import autocast, GradScaler

# Set random seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed()

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

# 1. Data preprocessing
def prepare_data(train_data, val_data):
    """
    Prepare data by combining title and abstract fields and extracting labels.
    
    Args:
        train_data (DataFrame): Training data
        val_data (DataFrame): Validation data
        
    Returns:
        tuple: train_texts, train_labels, val_texts, val_labels
    """
    # Combine title and abstract fields
    train_data['text'] = train_data[['title', 'abs']].fillna('').agg(' '.join, axis=1)
    val_data['text'] = val_data[['title', 'abs']].fillna('').agg(' '.join, axis=1)
    
    # Extract texts and labels
    train_texts = train_data['text'].tolist()
    train_labels = [int(label) for label in train_data['label'].tolist()]
    val_texts = val_data['text'].tolist()
    val_labels = [int(label) for label in val_data['label'].tolist()]
    
    return train_texts, train_labels, val_texts, val_labels

# 2. Custom dataset
class PatentDataset(Dataset):
    """
    Custom dataset for patent classification task that tokenizes text data.
    """
    def __init__(self, texts, labels, tokenizer, max_length=512):
        """
        Initialize dataset with texts, labels, and tokenizer.
        
        Args:
            texts (list): List of text samples
            labels (list): List of corresponding labels
            tokenizer: Tokenizer for encoding texts
            max_length (int): Maximum sequence length
        """
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
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors="pt"
        )
        item = {key: val.squeeze() for key, val in encoding.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.labels)

# 3. Training and evaluation functions
def train_epoch(model, train_loader, optimizer, scheduler, device, scaler=None, mixed_precision=False):
    """
    Train the model for one epoch with optional mixed precision.
    
    Args:
        model: The model to train
        train_loader: DataLoader for training data
        optimizer: Optimizer for parameter updates
        scheduler: Learning rate scheduler
        device: Device to train on
        scaler: GradScaler for mixed precision training
        mixed_precision (bool): Whether to use mixed precision training
        
    Returns:
        float: Average training loss for the epoch
    """
    model.train()
    total_loss = 0
    
    for batch in tqdm(train_loader, desc="Training"):
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        # Mixed precision training path
        if mixed_precision:
            with autocast():
                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                
                # Ensure loss is scalar
                if loss.dim() > 0:
                    loss = loss.mean()
            
            # Scale loss, perform backward pass, and update weights
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        
        # Standard training path (FP32)
        else:
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            
            # Ensure loss is scalar
            if loss.dim() > 0:
                loss = loss.mean()
                
            loss.backward()
            optimizer.step()
        
        scheduler.step()
        total_loss += loss.item()
        
    return total_loss / len(train_loader)

def evaluate_model(model, val_loader, device, mixed_precision=False):
    """
    Evaluate model and calculate Precision, Recall, F1-score, and Accuracy.
    
    Args:
        model: The model to evaluate
        val_loader: DataLoader for validation data
        device: Device to evaluate on
        mixed_precision (bool): Whether to use mixed precision
        
    Returns:
        dict: Dictionary of evaluation metrics
    """
    model.eval()
    true_labels = []
    predictions = []
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            # Use mixed precision for inference if enabled
            if mixed_precision:
                with autocast():
                    outputs = model(input_ids, attention_mask=attention_mask)
            else:
                outputs = model(input_ids, attention_mask=attention_mask)
                
            logits = outputs.logits
            predictions.extend(torch.argmax(logits, dim=1).cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

    # Calculate classification report
    report = classification_report(true_labels, predictions, output_dict=True, zero_division=0)
    
    # Average metrics across classes
    metrics = {
        'precision': (report['0']['precision'] + report['1']['precision']) / 2,
        'recall': (report['0']['recall'] + report['1']['recall']) / 2,
        'f1-score': (report['0']['f1-score'] + report['1']['f1-score']) / 2,
        'accuracy': accuracy_score(true_labels, predictions),
    }
    
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(classification_report(true_labels, predictions, zero_division=0))
    
    return metrics

# 4. Main training logic
def train_model(
    train_texts, 
    train_labels, 
    val_texts, 
    val_labels, 
    save_path, 
    num_epochs=20, 
    batch_size=16, 
    patience=3,
    mixed_precision=True
):
    """
    Train the model with early stopping based on validation F1-score.
    
    Args:
        train_texts (list): Training text samples
        train_labels (list): Training labels
        val_texts (list): Validation text samples
        val_labels (list): Validation labels
        save_path (str): Path to save the best model
        num_epochs (int): Maximum number of training epochs
        batch_size (int): Batch size for training
        patience (int): Early stopping patience
        mixed_precision (bool): Whether to use mixed precision training
        
    Returns:
        tuple: (best_val_f1, best_epoch)
    """
    # Initialize tokenizer for BERT base model
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    
    # Create datasets and data loaders
    train_dataset = PatentDataset(train_texts, train_labels, tokenizer)
    val_dataset = PatentDataset(val_texts, val_labels, tokenizer)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        collate_fn=data_collator,
        num_workers=4,  # Use multiple workers for data loading
        pin_memory=True  # Pin memory for faster data transfer to GPU
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        collate_fn=data_collator,
        num_workers=4,
        pin_memory=True
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        "bert-base-uncased", 
        num_labels=2
    )
    model.to(device)

    # Use DataParallel if multiple GPUs are available
    if torch.cuda.device_count() > 1:
        print(f"Detected {torch.cuda.device_count()} GPUs, enabling DataParallel mode")
        model = torch.nn.DataParallel(model)

    # Initialize optimizer
    optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    
    # Initialize learning rate scheduler
    total_steps = len(train_loader) * num_epochs
    scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )

    # Initialize gradient scaler for mixed precision training
    scaler = GradScaler() if mixed_precision else None

    # Initialize early stopping variables
    best_val_f1 = 0
    epochs_no_improve = 0
    best_epoch = -1

    # Training loop
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        
        # Train for one epoch
        train_loss = train_epoch(
            model, 
            train_loader, 
            optimizer, 
            scheduler, 
            device, 
            scaler=scaler, 
            mixed_precision=mixed_precision
        )
        print(f"Training Loss: {train_loss:.4f}")

        # Evaluate the model
        val_metrics = evaluate_model(
            model, 
            val_loader, 
            device, 
            mixed_precision=mixed_precision
        )

        # Check if this is the best model so far
        if val_metrics['f1-score'] > best_val_f1:
            best_val_f1 = val_metrics['f1-score']
            best_epoch = epoch + 1
            
            # Save the model
            if isinstance(model, torch.nn.DataParallel):
                model.module.save_pretrained(save_path)
            else:
                model.save_pretrained(save_path)
                
            tokenizer.save_pretrained(save_path)
            print(f"Best model saved to {save_path} at epoch {best_epoch} with F1-score: {best_val_f1:.4f}")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            print(f"No improvement for {epochs_no_improve} epochs.")

        # Early stopping check
        if epochs_no_improve >= patience:
            print("Early stopping triggered.")
            break

    print(f"Training completed. Best F1-score: {best_val_f1:.4f} achieved at epoch {best_epoch}.")
    return best_val_f1, best_epoch

# 5. Main function
if __name__ == "__main__":
    # Load data paths
    train_ai_path = './data/training/AI_training.csv'
    train_non_ai_path = './data/training/NonAI_training.csv'
    val_ai_path = './data/validation/AI_validation.csv'
    val_non_ai_path = './data/validation/NonAI_validation.csv'

    # Sample data for faster experimentation (adjust sample sizes as needed)
    train_ai = pd.read_csv(train_ai_path)
    train_non_ai = pd.read_csv(train_non_ai_path)
    val_ai = pd.read_csv(val_ai_path)
    val_non_ai = pd.read_csv(val_non_ai_path)

    # Merge and shuffle data
    train_data = pd.concat([train_ai, train_non_ai], ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)
    val_data = pd.concat([val_ai, val_non_ai], ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)

    # Check class distribution
    print("Training set class distribution:")
    print(train_data['label'].value_counts())
    print("Validation set class distribution:")
    print(val_data['label'].value_counts())

    # Prepare data
    train_texts, train_labels, val_texts, val_labels = prepare_data(train_data, val_data)

    # Ensure save path exists
    save_path = "./models/BERT/BestBERT"
    os.makedirs(save_path, exist_ok=True)

    # Train model with mixed precision
    train_model(
        train_texts=train_texts, 
        train_labels=train_labels, 
        val_texts=val_texts, 
        val_labels=val_labels, 
        save_path=save_path, 
        num_epochs=10, 
        batch_size=32,  # Increased batch size due to mixed precision efficiency
        patience=3,
        mixed_precision=True  # Enable mixed precision training
    )