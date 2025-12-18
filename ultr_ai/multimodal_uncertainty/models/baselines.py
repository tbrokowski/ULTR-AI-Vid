# =============================================================================
# Baseline Methods
# =============================================================================

import torch
import torch.nn as nn

from ultr_ai.multimodal_uncertainty.base import BaseMultimodalModel
from ultr_ai.multimodal_uncertainty.encoders import DomainAwareClinicalEncoder, SimpleClinicalEncoder

class SimpleBaselineModel(BaseMultimodalModel):
    """Simple baseline: raw concatenation without sophisticated processing"""
    
    def __init__(self, clinical_dim, embedding_dim, hidden_dim=256, dropout=0.1, 
                 feature_schema=None, feature_order=None):
        super().__init__(clinical_dim, embedding_dim, hidden_dim, dropout)
        
        # Intentionally ignore sophisticated feature processing for baseline
        self.classifier = nn.Sequential(
            nn.Linear(clinical_dim + embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, clinical_features, patient_embeddings):
        # Simple raw concatenation
        combined = torch.cat([clinical_features, patient_embeddings], dim=1)
        return self.classifier(combined)

# =============================================================================
# Non-Multimodal Baselines
# =============================================================================

class ClinicalOnlyModel(BaseMultimodalModel):
    """Baseline: Clinical features only"""
    
    def __init__(self, clinical_dim, embedding_dim, hidden_dim=256, dropout=0.1, 
                 feature_schema=None, feature_order=None):
        super().__init__(clinical_dim, embedding_dim, hidden_dim, dropout)
        
        # Use domain-aware encoder if available, otherwise simple
        if feature_schema and feature_order:
            self.clinical_encoder = DomainAwareClinicalEncoder(
                clinical_dim, hidden_dim, feature_schema, feature_order, hidden_dim, dropout
            )
            classifier_input = hidden_dim
        else:
            self.clinical_encoder = SimpleClinicalEncoder(clinical_dim, hidden_dim, hidden_dim, dropout)
            classifier_input = hidden_dim
        
        self.classifier = nn.Sequential(
            nn.Linear(classifier_input, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, clinical_features, patient_embeddings):
        # Ignore patient embeddings - clinical only
        clinical_encoded = self.clinical_encoder(clinical_features)
        return self.classifier(clinical_encoded)

class ImagingOnlyModel(BaseMultimodalModel):
    """Baseline: Imaging features only"""
    
    def __init__(self, clinical_dim, embedding_dim, hidden_dim=256, dropout=0.1, 
                 feature_schema=None, feature_order=None):
        super().__init__(clinical_dim, embedding_dim, hidden_dim, dropout)
        
        self.imaging_encoder = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, clinical_features, patient_embeddings):
        # Ignore clinical features - imaging only
        return self.imaging_encoder(patient_embeddings)