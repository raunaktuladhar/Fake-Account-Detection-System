"""
Improved fake profile detection model training script.
Addresses the issues in the original implementation.
"""
import sys
import pandas as pd
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import BertTokenizer, BertModel
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

def validate_data(df, required_columns):
    """Validate that the dataframe has required columns and data."""
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns: {missing_cols}")
    
    # Check for completely empty columns
    empty_cols = [col for col in required_columns if df[col].isna().all()]
    if empty_cols:
        logger.warning(f"Empty columns found: {empty_cols}")
    
    return True

def load_and_validate_data():
    """Load and validate the datasets."""
    required_columns = ["description", "followers_count", "friends_count", "statuses_count"]
    
    try:
        fusers = pd.read_csv("fusers.csv")
        users = pd.read_csv("users.csv")
        
        # Validate data
        validate_data(fusers, required_columns)
        validate_data(users, required_columns)
        
        logger.info(f"Loaded {len(fusers)} fake users and {len(users)} real users")
        
    except FileNotFoundError as e:
        logger.error(f"Error: {e}. Please ensure fusers.csv and users.csv are in the directory.")
        sys.exit(1)
    except ValueError as e:
        logger.error(f"Data validation error: {e}")
        sys.exit(1)
    
    # Add labels and combine
    fusers["label"] = 1  # Fake
    users["label"] = 0   # Real
    data = pd.concat([fusers, users], ignore_index=True)
    
    # Handle missing values more carefully
    data["description"] = data["description"].fillna("")
    data[["followers_count", "friends_count", "statuses_count"]] = data[
        ["followers_count", "friends_count", "statuses_count"]
    ].fillna(0)
    
    return data

def get_bert_embeddings_batch(descriptions, tokenizer, bert_model, batch_size=16):
    """Extract BERT embeddings in batches for efficiency."""
    embeddings = []
    
    with torch.no_grad():
        for i in tqdm(range(0, len(descriptions), batch_size), desc="Computing BERT embeddings"):
            batch_desc = descriptions[i:i+batch_size]
            batch_desc = [str(desc) if desc is not None else "" for desc in batch_desc]
            
            inputs = tokenizer(
                batch_desc, 
                return_tensors="pt", 
                truncation=True, 
                padding=True, 
                max_length=128
            )
            
            outputs = bert_model(**inputs)
            batch_embeddings = outputs.last_hidden_state.mean(dim=1)
            embeddings.append(batch_embeddings)
    
    return torch.cat(embeddings, dim=0).numpy()

class BertLSTM(nn.Module):
    """LSTM model that matches the training script architecture."""

    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        """Initializes the BertLSTM model - matches training script."""
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        """Forward pass - matches training script."""
        # Reshape for LSTM: (batch, seq_len, input_size)
        x = x.unsqueeze(1)
        h0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(x.device)
        c0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

class CombinedDataset(Dataset):
    """Dataset for combined text and numerical features."""
    
    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

def train_model(model, train_loader, val_loader, num_epochs=50, patience=10):
    """Train the model with proper validation and early stopping."""
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    
    best_val_loss = float('inf')
    patience_counter = 0
    train_losses = []
    val_losses = []
    val_accuracies = []
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_features, batch_labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            optimizer.zero_grad()
            outputs = model(batch_features)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += batch_labels.size(0)
            train_correct += (predicted == batch_labels).sum().item()
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch_features, batch_labels in val_loader:
                outputs = model(batch_features)
                loss = criterion(outputs, batch_labels)
                val_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                val_total += batch_labels.size(0)
                val_correct += (predicted == batch_labels).sum().item()
        
        # Calculate metrics
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        train_acc = 100 * train_correct / train_total
        val_acc = 100 * val_correct / val_total
        
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        val_accuracies.append(val_acc)
        
        logger.info(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, "
                   f"Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%")
        
        # Learning rate scheduling
        scheduler.step(avg_val_loss)
        
        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), "best_model.pt")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
    
    return train_losses, val_losses, val_accuracies

def evaluate_model(model, test_loader):
    """Evaluate the model and return detailed metrics."""
    model.eval()
    all_labels = []
    all_predictions = []
    all_probabilities = []
    
    with torch.no_grad():
        for batch_features, batch_labels in test_loader:
            outputs = model(batch_features)
            probabilities = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            
            all_labels.extend(batch_labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
    
    return all_labels, all_predictions, all_probabilities

def main():
    # Load and validate data
    logger.info("Loading and validating data...")
    data = load_and_validate_data()
    
    # Initialize models
    logger.info("Initializing BERT model...")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    bert_model = BertModel.from_pretrained("bert-base-uncased")
    bert_model.eval()  # Set to eval mode since we're not training it
    
    # Get BERT embeddings
    logger.info("Computing BERT embeddings...")
    bert_embeddings = get_bert_embeddings_batch(data["description"], tokenizer, bert_model)
    
    # Prepare numerical features with scaling
    logger.info("Preparing numerical features...")
    numerical_cols = ["followers_count", "friends_count", "statuses_count"]
    scaler = StandardScaler()
    numerical_features = scaler.fit_transform(data[numerical_cols].values)
    
    # Combine features
    features = np.hstack((bert_embeddings, numerical_features))
    labels = data["label"].values
    
    logger.info(f"Feature shape: {features.shape}, Labels shape: {labels.shape}")
    logger.info(f"Class distribution: Real: {np.sum(labels == 0)}, Fake: {np.sum(labels == 1)}")
    
    # Create datasets
    dataset = CombinedDataset(features, labels)
    
    # Split data: 70% train, 15% validation, 15% test
    total_size = len(dataset)
    train_size = int(0.7 * total_size)
    val_size = int(0.15 * total_size)
    test_size = total_size - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size]
    )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Initialize model
    model = BertLSTM(
        input_size=features.shape[1],
        hidden_size=64,
        num_layers=2,
        num_classes=2
    )
    
    logger.info(f"Model architecture:\n{model}")
    
    # Train model
    logger.info("Starting training...")
    train_losses, val_losses, val_accuracies = train_model(
        model, train_loader, val_loader, num_epochs=50, patience=10
    )
    
    # Load best model
    model.load_state_dict(torch.load("best_model.pt"))
    
    # Evaluate on test set
    logger.info("Evaluating on test set...")
    test_labels, test_predictions, test_probabilities = evaluate_model(model, test_loader)
    
    # Print detailed results
    accuracy = accuracy_score(test_labels, test_predictions)
    logger.info(f"Test Accuracy: {accuracy:.4f}")
    
    print("\nDetailed Classification Report:")
    print(classification_report(test_labels, test_predictions, 
                              target_names=['Real', 'Fake'], digits=4))
    
    # Confusion Matrix
    cm = confusion_matrix(test_labels, test_predictions)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Real', 'Fake'], yticklabels=['Real', 'Fake'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()
    
    # Plot training curves
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    
    plt.subplot(1, 3, 2)
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.title('Validation Accuracy')
    
    plt.tight_layout()
    plt.show()
    
    # Save final model and preprocessing components
    torch.save({
        'model_state_dict': model.state_dict(),
        'scaler': scaler,
        'feature_size': features.shape[1]
    }, 'final_model.pt')
    
    logger.info("Training completed successfully!")

if __name__ == "__main__":
    main()