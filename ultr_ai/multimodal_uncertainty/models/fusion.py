# =============================================================================
# Simple Fusion Methods
# =============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultr_ai.multimodal_uncertainty.base import BaseMultimodalModel
from ultr_ai.multimodal_uncertainty.encoders import DomainAwareClinicalEncoder, SimpleClinicalEncoder


class SimpleConcatenationModel(BaseMultimodalModel):
    """Simple concatenation of clinical and imaging features"""
    
    def __init__(self, clinical_dim, embedding_dim, hidden_dim=256, dropout=0.1, 
                 feature_schema=None, feature_order=None):
        super().__init__(clinical_dim, embedding_dim, hidden_dim, dropout)
        
        # Use domain-aware encoder if schema provided, otherwise simple encoder
        if feature_schema and feature_order:
            self.clinical_encoder = DomainAwareClinicalEncoder(
                clinical_dim, embedding_dim, feature_schema, feature_order, hidden_dim, dropout
            )
        else:
            self.clinical_encoder = SimpleClinicalEncoder(clinical_dim, embedding_dim, hidden_dim, dropout)
        
        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, clinical_features, patient_embeddings):
        clinical_encoded = self.clinical_encoder(clinical_features)
        combined = torch.cat([clinical_encoded, patient_embeddings], dim=1)
        return self.classifier(combined)

class EarlyFusionModel(BaseMultimodalModel):
    """Early fusion - process raw features together"""
    
    def __init__(self, clinical_dim, embedding_dim, hidden_dim=256, dropout=0.1, 
                 feature_schema=None, feature_order=None):
        super().__init__(clinical_dim, embedding_dim, hidden_dim, dropout)
        
        self.joint_encoder = nn.Sequential(
            nn.Linear(clinical_dim + embedding_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, clinical_features, patient_embeddings):
        combined = torch.cat([clinical_features, patient_embeddings], dim=1)
        return self.joint_encoder(combined)

class LateFusionModel(BaseMultimodalModel):
    """Late fusion - combine predictions from separate modalities"""
    
    def __init__(self, clinical_dim, embedding_dim, hidden_dim=256, dropout=0.1, 
                 feature_schema=None, feature_order=None):
        super().__init__(clinical_dim, embedding_dim, hidden_dim, dropout)
        
        # Separate predictors for each modality
        if feature_schema and feature_order:
            self.clinical_encoder = DomainAwareClinicalEncoder(
                clinical_dim, hidden_dim, feature_schema, feature_order, hidden_dim, dropout
            )
            self.clinical_predictor = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, 1)
            )
        else:
            self.clinical_predictor = nn.Sequential(
                nn.Linear(clinical_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, 1)
            )
            self.clinical_encoder = None
        
        self.imaging_predictor = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Learnable fusion weights
        self.fusion_weights = nn.Parameter(torch.tensor([0.5, 0.5]))
    
    def forward(self, clinical_features, patient_embeddings):
        if self.clinical_encoder:
            clinical_encoded = self.clinical_encoder(clinical_features)
            clinical_pred = self.clinical_predictor(clinical_encoded)
        else:
            clinical_pred = self.clinical_predictor(clinical_features)
            
        imaging_pred = self.imaging_predictor(patient_embeddings)
        
        # Weighted combination
        weights = F.softmax(self.fusion_weights, dim=0)
        fused_pred = weights[0] * clinical_pred + weights[1] * imaging_pred
        
        return fused_pred

# =============================================================================
# Advanced Fusion Methods
# =============================================================================

class ContrastiveFusionModel(BaseMultimodalModel):
    """Contrastive learning for multimodal alignment"""
    
    def __init__(self, clinical_dim, embedding_dim, hidden_dim=256, dropout=0.1, 
                 temperature=0.07, feature_schema=None, feature_order=None):
        super().__init__(clinical_dim, embedding_dim, hidden_dim, dropout)
        self.temperature = temperature
        
        # Advanced clinical processing
        if feature_schema and feature_order:
            self.clinical_processor = DomainAwareClinicalEncoder(
                clinical_dim, hidden_dim, feature_schema, feature_order, hidden_dim, dropout
            )
        else:
            self.clinical_processor = SimpleClinicalEncoder(clinical_dim, hidden_dim, hidden_dim, dropout)
        
        # Project both modalities to shared space
        self.clinical_projector = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embedding_dim)
        )
        
        self.imaging_projector = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embedding_dim)
        )
        
        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, clinical_features, patient_embeddings):
        clinical_processed = self.clinical_processor(clinical_features)
        clinical_proj = F.normalize(self.clinical_projector(clinical_processed), dim=1)
        imaging_proj = F.normalize(self.imaging_projector(patient_embeddings), dim=1)
        
        combined = torch.cat([clinical_proj, imaging_proj], dim=1)
        return self.classifier(combined)
    
    def contrastive_loss(self, clinical_features, patient_embeddings, labels):
        """Compute contrastive loss for multimodal alignment"""
        clinical_processed = self.clinical_processor(clinical_features)
        clinical_proj = F.normalize(self.clinical_projector(clinical_processed), dim=1)
        imaging_proj = F.normalize(self.imaging_projector(patient_embeddings), dim=1)
        
        # Compute similarity matrix
        similarity = torch.mm(clinical_proj, imaging_proj.t()) / self.temperature
        
        # Create positive pairs mask (same patient)
        batch_size = clinical_features.size(0)
        mask = torch.eye(batch_size, device=clinical_features.device)
        
        # Compute contrastive loss
        exp_sim = torch.exp(similarity)
        pos_sim = exp_sim * mask
        neg_sim = exp_sim * (1 - mask)
        
        loss = -torch.log(pos_sim.sum(dim=1) / (pos_sim.sum(dim=1) + neg_sim.sum(dim=1) + 1e-8))
        return loss.mean()

class CrossAttentionFusionModel(BaseMultimodalModel):
    """Cross-attention between clinical and imaging features"""
    
    def __init__(self, clinical_dim, embedding_dim, hidden_dim=256, dropout=0.1, 
                 num_heads=8, feature_schema=None, feature_order=None):
        super().__init__(clinical_dim, embedding_dim, hidden_dim, dropout)
        
        # Advanced clinical processing
        if feature_schema and feature_order:
            self.clinical_encoder = DomainAwareClinicalEncoder(
                clinical_dim, hidden_dim, feature_schema, feature_order, hidden_dim, dropout
            )
        else:
            self.clinical_encoder = SimpleClinicalEncoder(clinical_dim, hidden_dim, hidden_dim, dropout)
        
        self.imaging_encoder = nn.Linear(embedding_dim, hidden_dim)
        
        # Cross-attention layers
        self.clinical_to_imaging_attention = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.imaging_to_clinical_attention = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        
        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, clinical_features, patient_embeddings):
        clinical_encoded = self.clinical_encoder(clinical_features).unsqueeze(1)  # Add sequence dim
        imaging_encoded = self.imaging_encoder(patient_embeddings).unsqueeze(1)
        
        # Cross-attention
        clinical_attended, _ = self.clinical_to_imaging_attention(
            clinical_encoded, imaging_encoded, imaging_encoded
        )
        imaging_attended, _ = self.imaging_to_clinical_attention(
            imaging_encoded, clinical_encoded, clinical_encoded
        )
        
        # Remove sequence dimension and concatenate
        clinical_attended = clinical_attended.squeeze(1)
        imaging_attended = imaging_attended.squeeze(1)
        
        combined = torch.cat([clinical_attended, imaging_attended], dim=1)
        return self.classifier(combined)

class BilinearFusionModel(BaseMultimodalModel):
    """Bilinear fusion with advanced clinical processing"""
    
    def __init__(self, clinical_dim, embedding_dim, hidden_dim=256, dropout=0.1, 
                 feature_schema=None, feature_order=None):
        super().__init__(clinical_dim, embedding_dim, hidden_dim, dropout)
        
        # Advanced clinical processing
        if feature_schema and feature_order:
            self.clinical_encoder = DomainAwareClinicalEncoder(
                clinical_dim, hidden_dim, feature_schema, feature_order, hidden_dim, dropout
            )
        else:
            self.clinical_encoder = SimpleClinicalEncoder(clinical_dim, hidden_dim, hidden_dim, dropout)
        
        self.imaging_encoder = nn.Linear(embedding_dim, hidden_dim)
        
        # Bilinear fusion layer
        self.bilinear = nn.Bilinear(hidden_dim, hidden_dim, hidden_dim)
        
        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, clinical_features, patient_embeddings):
        clinical_encoded = self.clinical_encoder(clinical_features)
        imaging_encoded = self.imaging_encoder(patient_embeddings)
        
        # Bilinear interaction
        fused = self.bilinear(clinical_encoded, imaging_encoded)
        return self.classifier(fused)

# =============================================================================
# Novel Fusion Method: Clinical Contextual Gating (CCG)
# =============================================================================

class ClinicalContextualGating(BaseMultimodalModel):
    """Novel: Dynamic gating of imaging features based on clinical context"""
    
    def __init__(self, clinical_dim, embedding_dim, hidden_dim=256, dropout=0.1, 
                 feature_schema=None, feature_order=None):
        super().__init__(clinical_dim, embedding_dim, hidden_dim, dropout)
        
        # Advanced clinical processing for context and encoding
        if feature_schema and feature_order:
            self.clinical_context_encoder = DomainAwareClinicalEncoder(
                clinical_dim, hidden_dim // 2, feature_schema, feature_order, hidden_dim, dropout
            )
            self.clinical_encoder = DomainAwareClinicalEncoder(
                clinical_dim, embedding_dim, feature_schema, feature_order, hidden_dim, dropout
            )
        else:
            self.clinical_context_encoder = SimpleClinicalEncoder(
                clinical_dim, hidden_dim // 2, hidden_dim, dropout
            )
            self.clinical_encoder = SimpleClinicalEncoder(
                clinical_dim, embedding_dim, hidden_dim, dropout
            )
        
        # Dynamic gating network
        self.gating_network = nn.Sequential(
            nn.Linear(hidden_dim // 2, embedding_dim),
            nn.Sigmoid()  # Gates between 0 and 1
        )
        
        # Uncertainty-aware gates (additional confidence)
        self.confidence_network = nn.Sequential(
            nn.Linear(hidden_dim // 2, embedding_dim),
            nn.Sigmoid()
        )
        
        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, clinical_features, patient_embeddings):
        # Encode clinical context
        clinical_context = self.clinical_context_encoder(clinical_features)
        
        # Generate gates based on clinical context
        gates = self.gating_network(clinical_context)
        confidence = self.confidence_network(clinical_context)
        
        # Apply uncertainty-aware gating to imaging features
        gated_imaging = patient_embeddings * gates * confidence
        
        # Encode clinical features
        clinical_encoded = self.clinical_encoder(clinical_features)
        
        # Combine
        combined = torch.cat([clinical_encoded, gated_imaging], dim=1)
        return self.classifier(combined)
    
    def get_gate_attention(self, clinical_features):
        """Return gate weights for interpretability"""
        clinical_context = self.clinical_context_encoder(clinical_features)
        gates = self.gating_network(clinical_context)
        confidence = self.confidence_network(clinical_context)
        return gates, confidence