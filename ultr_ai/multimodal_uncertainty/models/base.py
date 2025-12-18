from abc import ABC, abstractmethod
import torch.nn as nn

class BaseMultimodalModel(nn.Module, ABC):
    """Abstract base class for multimodal TB classification models"""
    
    def __init__(self, clinical_dim, embedding_dim, hidden_dim=256, dropout=0.1):
        super().__init__()
        self.clinical_dim = clinical_dim
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        
    @abstractmethod
    def forward(self, clinical_features, patient_embeddings):
        pass