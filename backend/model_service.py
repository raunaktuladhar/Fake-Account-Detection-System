"""
Fixed model service for fake profile detection.
Compatible with the training script architecture.
"""
import torch
from torch import nn
from transformers import BertTokenizer, BertModel
import numpy as np
import logging
import os
from typing import Dict, Any
import pickle

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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


class ImprovedFakeProfileClassifier(nn.Module):
    """
    Alternative classifier that matches the improved training script.
    Use this if you trained with the improved version.
    """
    
    def __init__(self, input_size, hidden_sizes=[128, 64, 32], num_classes=2, dropout_rate=0.3):
        super().__init__()
        
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_size = hidden_size
        
        layers.append(nn.Linear(prev_size, num_classes))
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)


class ModelService:
    """Service to load and use the trained model for predictions."""

    def __init__(self, model_type="original", model_path="./bert_lstm_model.pt", 
                 bert_model_path="./bert_model", scaler_path=None):
        """
        Initialize the service and load models.
        
        Args:
            model_type: "original" for BertLSTM, "improved" for ImprovedFakeProfileClassifier
            model_path: Path to the trained model file
            bert_model_path: Path to the BERT model directory
            scaler_path: Path to the scaler (for improved model)
        """
        self.model_type = model_type
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        try:
            self._load_models(model_path, bert_model_path, scaler_path)
            logger.info("Models loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load models: {e}")
            raise

    def _load_models(self, model_path, bert_model_path, scaler_path):
        """Load BERT tokenizer, model, and classifier."""
        
        # Load BERT components
        if os.path.exists(bert_model_path):
            self.tokenizer = BertTokenizer.from_pretrained(bert_model_path)
            self.bert_model = BertModel.from_pretrained(bert_model_path)
        else:
            logger.warning("Local BERT model not found, using pre-trained")
            self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
            self.bert_model = BertModel.from_pretrained("bert-base-uncased")
        
        self.bert_model.eval()
        self.bert_model.to(self.device)
        
        # Freeze BERT parameters
        for param in self.bert_model.parameters():
            param.requires_grad = False
        
        # Load scaler if provided (for improved model)
        self.scaler = None
        if scaler_path and os.path.exists(scaler_path):
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
        
        # Load classifier model
        if self.model_type == "improved":
            self._load_improved_model(model_path)
        else:
            self._load_original_model(model_path)

    def _load_original_model(self, model_path):
        """Load the original BertLSTM model."""
        # Calculate input size: BERT embeddings (768) + numerical features (3)
        input_size = 768 + 3
        
        self.classifier = BertLSTM(
            input_size=input_size,
            hidden_size=64,
            num_layers=2,
            num_classes=2
        )
        
        if os.path.exists(model_path):
            self.classifier.load_state_dict(torch.load(model_path, map_location=self.device))
        else:
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        self.classifier.eval()
        self.classifier.to(self.device)

    def _load_improved_model(self, model_path):
        """Load the improved feedforward model."""
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=self.device)
            
            if isinstance(checkpoint, dict) and 'feature_size' in checkpoint:
                # New format with metadata
                feature_size = checkpoint['feature_size']
                self.classifier = ImprovedFakeProfileClassifier(input_size=feature_size)
                self.classifier.load_state_dict(checkpoint['model_state_dict'])
                
                # Load scaler if included
                if 'scaler' in checkpoint:
                    self.scaler = checkpoint['scaler']
            else:
                # Old format - assume standard size
                feature_size = 768 + 3  # BERT + numerical features
                self.classifier = ImprovedFakeProfileClassifier(input_size=feature_size)
                self.classifier.load_state_dict(checkpoint)
        else:
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        self.classifier.eval()
        self.classifier.to(self.device)

    def _extract_bert_embeddings(self, description: str) -> np.ndarray:
        """Extract BERT embeddings for a single description."""
        description = str(description) if description else ""
        
        inputs = self.tokenizer(
            description,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128
        )
        
        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.bert_model(**inputs)
            # Use mean pooling of last hidden state
            embeddings = outputs.last_hidden_state.mean(dim=1)
        
        return embeddings.cpu().numpy()

    def _prepare_features(self, data: Dict[str, Any]) -> torch.Tensor:
        """Prepare features for model input."""
        # Extract text features
        description = data.get("description", "")
        bert_embeddings = self._extract_bert_embeddings(description)
        
        # Extract numerical features
        numerical_features = np.array([
            float(data.get("followers_count", 0)),
            float(data.get("friends_count", 0)),
            float(data.get("statuses_count", 0)),
        ]).reshape(1, -1)
        
        # Apply scaling if available
        if self.scaler is not None:
            numerical_features = self.scaler.transform(numerical_features)
        
        # Combine features
        combined_features = np.hstack((bert_embeddings, numerical_features))
        
        return torch.tensor(combined_features, dtype=torch.float32).to(self.device)

    def predict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predict if a profile is fake or real.
        
        Args:
            data: Dictionary containing profile information with keys:
                  - description: Profile description text
                  - followers_count: Number of followers
                  - friends_count: Number of friends/following
                  - statuses_count: Number of status updates
        
        Returns:
            Dictionary with prediction results
        """
        try:
            # Validate input data
            if not isinstance(data, dict):
                raise ValueError("Input data must be a dictionary")
            
            # Prepare features
            features = self._prepare_features(data)
            
            # Make prediction
            with torch.no_grad():
                outputs = self.classifier(features)
                
                # Apply temperature scaling to soften probabilities
                temperature = 5.0  # Higher temp -> softer probabilities
                scaled_outputs = outputs / temperature
                probabilities = torch.softmax(scaled_outputs, dim=1)
                predicted_class = probabilities.argmax(dim=1).item()
                confidence = probabilities[0][predicted_class].item() * 100
                
                # Get probabilities for both classes
                real_prob = probabilities[0][0].item() * 100
                fake_prob = probabilities[0][1].item() * 100
            
            return {
                "prediction": "real" if predicted_class == 0 else "fake",
                "confidence": {
                    "real": real_prob,
                    "fake": fake_prob
                }
            }
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return {
                "prediction": "error",
                "confidence": 0.0,
                "error": str(e),
                "probabilities": {"real": 0.0, "fake": 0.0}
            }

    def predict_batch(self, data_list: list) -> list:
        """Predict for multiple profiles at once."""
        results = []
        for data in data_list:
            result = self.predict(data)
            results.append(result)
        return results

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        return {
            "model_type": self.model_type,
            "device": str(self.device),
            "has_scaler": self.scaler is not None,
            "bert_model": self.bert_model.config.name_or_path if hasattr(self.bert_model.config, 'name_or_path') else "unknown"
        }


# Example usage
if __name__ == "__main__":
    # Test with original model
    try:
        service = ModelService(model_type="original")
        
        # Test data
        test_profile = {
            "description": "Love traveling and photography! Follow me for amazing pics!",
            "followers_count": 1500,
            "friends_count": 800,
            "statuses_count": 250
        }
        
        result = service.predict(test_profile)
        print("Prediction Result:", result)
        print("Model Info:", service.get_model_info())
        
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure the model files exist and are properly trained.")