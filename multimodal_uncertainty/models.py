import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional, Union

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

# =============================================================================
# Advanced Clinical Encoders with Domain Awareness
# =============================================================================
class DomainAwareClinicalEncoder(nn.Module):
    """FIXED: Compatible clinical encoder that matches models.py interface exactly"""
    
    def __init__(self, clinical_dim: int, output_dim: int, feature_schema: Dict[str, List[str]], 
                 feature_order: List[str], hidden_dim: int = 256, dropout: float = 0.1, 
                 use_attention: bool = True, use_batch_norm: bool = True):
        super().__init__()
        
        self.clinical_dim = clinical_dim
        self.output_dim = output_dim
        self.feature_schema = feature_schema if feature_schema else {}
        self.feature_order = feature_order if feature_order else []
        self.use_attention = use_attention
        self.use_batch_norm = use_batch_norm
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        
        # Always create a simple encoder as fallback
        self.simple_encoder = self._build_simple_encoder(clinical_dim, output_dim, dropout)
        
        # Validate inputs and decide on encoder type
        if self._validate_domain_setup():
            self.use_domain_aware = True
            self._build_domain_encoders()
            print(f"✅ Domain-aware encoder initialized with {len(self.domain_encoders)} domains")
        else:
            print(f"⚠️ Domain setup invalid. Using simple encoder fallback.")
            self.use_domain_aware = False
    
    def _validate_domain_setup(self) -> bool:
        """Validate that domain setup is correct"""
        if not self.feature_schema or not self.feature_order:
            return False
        
        # Check if we can map features to indices
        try:
            self.domain_indices = self._create_domain_indices()
            total_features = sum(len(indices) for indices in self.domain_indices.values())
            
            # Allow some tolerance for mismatched features
            if abs(total_features - self.clinical_dim) > 5:  # Allow 5 feature difference
                print(f"Feature mismatch: Expected {self.clinical_dim}, mapped {total_features}")
                return False
            return True
        except Exception as e:
            print(f"Domain validation failed: {e}")
            return False
    
    def _create_domain_indices(self) -> Dict[str, List[int]]:
        """Create mapping from domain names to feature indices"""
        domain_indices = {}
        
        for domain, features in self.feature_schema.items():
            indices = []
            for feature in features:
                if feature in self.feature_order:
                    indices.append(self.feature_order.index(feature))
            if indices:  # Only add domains that have features
                domain_indices[domain] = indices
        
        return domain_indices
    
    def _build_domain_encoders(self):
        """Build domain-specific encoders"""
        self.domain_encoders = nn.ModuleDict()
        
        # Define domain-specific architectures - compatible with original
        domain_configs = {
            'demographics': {'layers': 2, 'output_dim': max(8, self.hidden_dim // 8)},
            'symptoms': {'layers': 3, 'output_dim': max(32, self.hidden_dim // 2)},
            'vital_signs': {'layers': 2, 'output_dim': max(16, self.hidden_dim // 4)},
            'medical_history': {'layers': 2, 'output_dim': max(16, self.hidden_dim // 4)},
            'physical_exam': {'layers': 2, 'output_dim': max(8, self.hidden_dim // 8)},
            'laboratory': {'layers': 3, 'output_dim': max(16, self.hidden_dim // 4)},
            'management': {'layers': 2, 'output_dim': max(8, self.hidden_dim // 8)},
            'other': {'layers': 2, 'output_dim': max(8, self.hidden_dim // 8)}
        }
        
        total_encoded_dim = 0
        
        for domain, indices in self.domain_indices.items():
            if indices:  # Only create encoder if domain has features
                config = domain_configs.get(domain, {'layers': 2, 'output_dim': max(8, self.hidden_dim // 4)})
                input_dim = len(indices)
                
                # Ensure output_dim is not larger than input_dim for very small domains
                output_dim = min(config['output_dim'], max(4, input_dim))
                
                try:
                    self.domain_encoders[domain] = self._build_encoder(
                        input_dim, 
                        output_dim, 
                        self.dropout, 
                        config['layers']
                    )
                    total_encoded_dim += output_dim
                except Exception as e:
                    print(f"Failed to create encoder for domain {domain}: {e}")
                    continue
        
        # Cross-domain attention mechanism
        if self.use_attention and len(self.domain_encoders) > 1:
            try:
                attention_dim = max(16, self.hidden_dim // 4)
                self.domain_attention = nn.MultiheadAttention(
                    embed_dim=attention_dim, num_heads=min(4, max(1, attention_dim // 4)), 
                    dropout=self.dropout, batch_first=True
                )
                self.attention_norm = nn.LayerNorm(attention_dim)
                
                # Projection layers to common dimension for attention
                self.domain_projections = nn.ModuleDict()
                for domain in self.domain_encoders.keys():
                    config = domain_configs.get(domain, {'output_dim': max(8, self.hidden_dim // 4)})
                    domain_output_dim = min(config['output_dim'], max(4, len(self.domain_indices[domain])))
                    self.domain_projections[domain] = nn.Linear(domain_output_dim, attention_dim)
                
                # Update total dimension to include attention output
                total_encoded_dim += attention_dim
            except Exception as e:
                print(f"Failed to create attention mechanism: {e}")
                self.use_attention = False
        
        # Final fusion layer - ensure reasonable input size
        if total_encoded_dim > 0:
            try:
                self.fusion_layer = self._build_encoder(
                    total_encoded_dim, self.output_dim, self.dropout, layers=3, final_activation=False
                )
                print(f"✅ Fusion layer created: {total_encoded_dim} -> {self.output_dim}")
            except Exception as e:
                print(f"Failed to create fusion layer: {e}")
                # Fallback if fusion fails
                self.use_domain_aware = False
        else:
            # Fallback if no domains created
            print("No domain encoders created, using simple encoder")
            self.use_domain_aware = False
    
    def _build_simple_encoder(self, input_dim: int, output_dim: int, dropout: float) -> nn.Module:
        """Build simple encoder as fallback"""
        if input_dim <= 0:
            return nn.Identity()
            
        # Handle case where input_dim is very small
        hidden_dim = min(self.hidden_dim, max(32, input_dim * 2))
        intermediate_dim = min(hidden_dim // 2, max(16, input_dim))
        
        return nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim) if self.use_batch_norm else nn.Identity(),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, intermediate_dim),
            nn.BatchNorm1d(intermediate_dim) if self.use_batch_norm else nn.Identity(),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(intermediate_dim, output_dim)
        )
    
    def _build_encoder(self, input_dim: int, output_dim: int, dropout: float, 
                      layers: int = 3, final_activation: bool = True) -> nn.Module:
        """Build a flexible encoder with configurable layers"""
        if input_dim == 0:
            return nn.Identity()
        
        if input_dim <= 2:  # Very small input
            return nn.Linear(input_dim, output_dim)
            
        modules = []
        current_dim = input_dim
        
        for i in range(layers - 1):
            next_dim = max(output_dim, current_dim // 2)
            if next_dim >= current_dim:  # Prevent dimension increase
                next_dim = max(output_dim, current_dim // 2)
            
            modules.extend([
                nn.Linear(current_dim, next_dim),
                nn.BatchNorm1d(next_dim) if self.use_batch_norm else nn.Identity(),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            current_dim = next_dim
        
        # Final layer
        modules.append(nn.Linear(current_dim, output_dim))
        if final_activation:
            if self.use_batch_norm:
                modules.append(nn.BatchNorm1d(output_dim))
            modules.extend([nn.ReLU(), nn.Dropout(dropout)])
        
        return nn.Sequential(*modules)
    
    def forward(self, clinical_features: torch.Tensor) -> torch.Tensor:
        # Input validation and NaN handling
        if torch.isnan(clinical_features).any() or torch.isinf(clinical_features).any():
            print("Warning: NaN/Inf values detected in clinical features, replacing with zeros")
            clinical_features = torch.nan_to_num(clinical_features, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # Handle batch norm issues for single samples
        is_single_sample = clinical_features.size(0) == 1
        if is_single_sample and self.use_batch_norm:
            # Set to eval mode for single samples to avoid batch norm issues
            original_training = self.training
            self.eval()
        
        try:
            if not self.use_domain_aware:
                result = self.simple_encoder(clinical_features)
            else:
                result = self._forward_domain_aware(clinical_features)
        except Exception as e:
            print(f"Error in encoder forward pass: {e}")
            # Final fallback
            result = self.simple_encoder(clinical_features)
        finally:
            # Restore training mode if changed
            if is_single_sample and self.use_batch_norm:
                self.train(original_training)
        
        # Final check for NaN output
        if torch.isnan(result).any():
            print("Warning: NaN detected in encoder output, replacing with zeros")
            result = torch.nan_to_num(result, nan=0.0)
        
        return result
    
    def _forward_domain_aware(self, clinical_features: torch.Tensor) -> torch.Tensor:
        """Domain-aware forward pass"""
        # Split features by domain and encode
        domain_outputs = {}
        
        for domain, indices in self.domain_indices.items():
            if indices and domain in self.domain_encoders:
                domain_features = clinical_features[:, indices]
                
                try:
                    domain_outputs[domain] = self.domain_encoders[domain](domain_features)
                except Exception as e:
                    print(f"Error in domain '{domain}' encoder: {e}")
                    continue
        
        if not domain_outputs:
            # Fallback if no domain encoders worked
            print("Warning: No domain encoders produced output, using simple encoder")
            raise ValueError("Domain encoding failed")
        
        # Concatenate domain outputs
        encoded_features = []
        attention_features = []
        
        for domain, output in domain_outputs.items():
            encoded_features.append(output)
            
            # Prepare for attention if enabled
            if self.use_attention and domain in self.domain_projections:
                try:
                    projected = self.domain_projections[domain](output)
                    attention_features.append(projected)
                except Exception as e:
                    print(f"Error in attention projection for domain '{domain}': {e}")
                    continue
        
        # Apply cross-domain attention
        if self.use_attention and len(attention_features) > 1:
            try:
                # Stack for attention: (batch, num_domains, hidden_dim//4)
                attention_input = torch.stack(attention_features, dim=1)
                
                # Self-attention across domains
                attended, _ = self.domain_attention(
                    attention_input, attention_input, attention_input
                )
                attended = self.attention_norm(attended + attention_input)
                
                # Pool attention output (mean across domains)
                attention_output = attended.mean(dim=1)
                encoded_features.append(attention_output)
            except Exception as e:
                print(f"Error in cross-domain attention: {e}")
                # Continue without attention
        
        # Final fusion
        combined = torch.cat(encoded_features, dim=1)
        result = self.fusion_layer(combined)
        
        return result


class SimpleClinicalEncoder(nn.Module):
    """FIXED: Simple clinical encoder compatible with models.py"""
    
    def __init__(self, clinical_dim: int, output_dim: int, hidden_dim: int = 256, 
                 dropout: float = 0.1, use_batch_norm: bool = True):
        super().__init__()
        
        self.clinical_dim = clinical_dim
        self.output_dim = output_dim
        self.use_batch_norm = use_batch_norm
        
        if clinical_dim <= 0:
            self.encoder = nn.Identity()
        else:
            self.encoder = self._build_encoder(clinical_dim, output_dim, hidden_dim, dropout)
    
    def _build_encoder(self, input_dim: int, output_dim: int, hidden_dim: int, dropout: float) -> nn.Module:
        """Build a simple progressive encoder"""
        if input_dim <= 0:
            return nn.Identity()
        
        # Adjust hidden dimensions based on input size
        hidden_dim = min(hidden_dim, max(32, input_dim * 2))
        intermediate_dim = min(hidden_dim // 2, max(16, input_dim))
        
        modules = []
        
        # Progressive reduction
        dims = [input_dim, hidden_dim, intermediate_dim, output_dim]
        
        for i in range(len(dims) - 1):
            modules.append(nn.Linear(dims[i], dims[i + 1]))
            
            # Add batch norm and activation for all but the final layer
            if i < len(dims) - 2:
                if self.use_batch_norm:
                    modules.append(nn.BatchNorm1d(dims[i + 1]))
                modules.append(nn.ReLU())
                modules.append(nn.Dropout(dropout))
        
        return nn.Sequential(*modules)
    
    def forward(self, clinical_features: torch.Tensor) -> torch.Tensor:
        # Input validation and NaN handling
        if torch.isnan(clinical_features).any() or torch.isinf(clinical_features).any():
            clinical_features = torch.nan_to_num(clinical_features, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # Handle batch norm issues for single samples
        is_single_sample = clinical_features.size(0) == 1
        if is_single_sample and self.use_batch_norm:
            original_training = self.training
            self.eval()
        
        try:
            result = self.encoder(clinical_features)
        finally:
            if is_single_sample and self.use_batch_norm:
                self.train(original_training)
        
        # Output validation
        if torch.isnan(result).any():
            result = torch.nan_to_num(result, nan=0.0)
        
        return result


def test_encoder_compatibility():
    """Test encoder compatibility with typical inputs"""
    print("Testing encoder compatibility...")
    
    # Test parameters
    batch_size = 16
    clinical_dim = 50
    output_dim = 128
    
    # Create test data
    clinical_features = torch.randn(batch_size, clinical_dim)
    
    # Test simple encoder
    simple_encoder = SimpleClinicalEncoder(clinical_dim, output_dim)
    simple_output = simple_encoder(clinical_features)
    print(f"Simple encoder output shape: {simple_output.shape}")
    
    # Test domain-aware encoder with mock schema
    feature_names = [f'feature_{i}' for i in range(clinical_dim)]
    feature_schema = {
        'demographics': feature_names[:5],
        'symptoms': feature_names[5:20],
        'vital_signs': feature_names[20:30],
        'laboratory': feature_names[30:40],
        'other': feature_names[40:]
    }
    
    domain_encoder = DomainAwareClinicalEncoder(
        clinical_dim=clinical_dim,
        output_dim=output_dim,
        feature_schema=feature_schema,
        feature_order=feature_names
    )
    
    domain_output = domain_encoder(clinical_features)
    print(f"Domain-aware encoder output shape: {domain_output.shape}")
    
    # Test with problematic inputs
    zero_features = torch.zeros(batch_size, clinical_dim)
    zero_output = domain_encoder(zero_features)
    print(f"Zero features output shape: {zero_output.shape}")
    
    # Test with NaN inputs
    nan_features = clinical_features.clone()
    nan_features[0, :5] = float('nan')
    nan_output = domain_encoder(nan_features)
    print(f"NaN features output shape: {nan_output.shape}")
    
    print("✅ All compatibility tests passed!")
    return True


# if __name__ == "__main__":
#     test_encoder_compatibility()
# =============================================================================
# Baseline Methods
# =============================================================================

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

# =============================================================================
# Simple Fusion Methods
# =============================================================================

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

# =============================================================================
# Advanced Uncertainty Estimation Methods
# =============================================================================

class EvidentialModel(BaseMultimodalModel):
    """Evidential Deep Learning for uncertainty estimation"""
    
    def __init__(self, clinical_dim, embedding_dim, hidden_dim=256, dropout=0.1, 
                 feature_schema=None, feature_order=None):
        super().__init__(clinical_dim, embedding_dim, hidden_dim, dropout)
        
        # Advanced clinical processing
        if feature_schema and feature_order:
            self.clinical_encoder = DomainAwareClinicalEncoder(
                clinical_dim, embedding_dim, feature_schema, feature_order, hidden_dim, dropout
            )
        else:
            self.clinical_encoder = SimpleClinicalEncoder(clinical_dim, embedding_dim, hidden_dim, dropout)
        
        # Evidential network - outputs Dirichlet parameters
        self.evidential_head = nn.Sequential(
            nn.Linear(embedding_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 4)  # [alpha0, alpha1, beta, lambda] for Dirichlet
        )
        
    def forward(self, clinical_features, patient_embeddings):
        clinical_encoded = self.clinical_encoder(clinical_features)
        combined = torch.cat([clinical_encoded, patient_embeddings], dim=1)
        
        # Get Dirichlet parameters
        dirichlet_params = self.evidential_head(combined)
        
        # Split parameters: alpha (concentration), beta (precision), lambda (regularization)
        alpha = F.softplus(dirichlet_params[:, :2]) + 1  # Ensure alpha > 1
        beta = F.softplus(dirichlet_params[:, 2:3]) + 1   # Precision parameter
        
        # Dirichlet strength (evidence)
        S = torch.sum(alpha, dim=1, keepdim=True)
        
        # Predicted probability (expectation of Dirichlet)
        prob = alpha / S
        
        # Epistemic uncertainty (mutual information)
        epistemic_uncertainty = torch.sum(alpha * (torch.digamma(S) - torch.digamma(alpha)), dim=1, keepdim=True)
        
        # Aleatoric uncertainty (entropy of expected categorical)
        aleatoric_uncertainty = -torch.sum(prob * torch.log(prob + 1e-8), dim=1, keepdim=True)
        
        # Total uncertainty
        total_uncertainty = epistemic_uncertainty + aleatoric_uncertainty
        
        # Return logits for binary classification (use class 1 probability)
        logits = torch.log(prob[:, 1:2] / (prob[:, 0:1] + 1e-8))
        
        return logits, {
            'dirichlet_alpha': alpha,
            'evidence_strength': S,
            'epistemic_uncertainty': epistemic_uncertainty,
            'aleatoric_uncertainty': aleatoric_uncertainty,
            'total_uncertainty': total_uncertainty,
            'class_probabilities': prob
        }
    
    def evidential_loss(self, logits, targets, uncertainty_dict, epoch=0, annealing_coef=1.0):
        """Evidential loss with KL regularization"""
        alpha = uncertainty_dict['dirichlet_alpha']
        S = uncertainty_dict['evidence_strength']
        
        # Convert targets to one-hot
        targets_onehot = torch.zeros_like(alpha)
        targets_onehot.scatter_(1, targets.long(), 1)
        
        # Likelihood loss (negative log-likelihood of Dirichlet-categorical)
        likelihood_loss = torch.sum(targets_onehot * (torch.digamma(S) - torch.digamma(alpha)), dim=1)
        
        # KL regularization (encourage low evidence for incorrect predictions)
        kl_alpha = (alpha - 1) * (1 - targets_onehot) + 1
        kl_loss = torch.sum((alpha - kl_alpha) * (torch.digamma(alpha) - torch.digamma(S)), dim=1)
        
        # Total loss with annealing
        total_loss = likelihood_loss + annealing_coef * kl_loss
        
        return total_loss.mean()

class MonteCarloFrequencyModel(nn.Module):
    """Monte Carlo Frequency Analysis for advanced uncertainty estimation"""
    
    def __init__(self, base_model, dropout_rate=0.1, num_frequencies=50):
        super().__init__()
        self.base_model = base_model
        self.dropout_rate = dropout_rate
        self.num_frequencies = num_frequencies
        self.clinical_dim = base_model.clinical_dim
        self.embedding_dim = base_model.embedding_dim
        
        # Replace dropouts for MC sampling
        self._replace_dropouts(self.base_model)
        
        # Frequency analysis components
        self.frequency_analyzer = nn.Sequential(
            nn.Linear(num_frequencies, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 3)  # [frequency_uncertainty, spectral_density, coherence]
        )
    
    def _replace_dropouts(self, module):
        """Replace standard dropout with MC dropout"""
        for name, child in module.named_children():
            if isinstance(child, nn.Dropout):
                setattr(module, name, nn.Dropout(p=self.dropout_rate))
            else:
                self._replace_dropouts(child)
    
    def forward(self, clinical_features, patient_embeddings):
        return self.base_model(clinical_features, patient_embeddings)
    
    def _compute_frequency_features(self, predictions):
        """Compute frequency domain features from MC predictions"""
        # predictions shape: [num_samples, batch_size, 1]
        predictions = predictions.squeeze(-1)  # [num_samples, batch_size]
        
        # Compute FFT along the MC sample dimension
        fft_preds = torch.fft.fft(predictions, dim=0, n=self.num_frequencies)
        
        # Power spectral density
        psd = torch.abs(fft_preds) ** 2
        
        # Spectral centroid (center of mass of spectrum)
        freqs = torch.arange(self.num_frequencies, device=predictions.device, dtype=torch.float32)
        spectral_centroid = torch.sum(freqs.unsqueeze(1) * psd, dim=0) / (torch.sum(psd, dim=0) + 1e-8)
        
        # Spectral rolloff (frequency below which 85% of energy is contained)
        cumsum_psd = torch.cumsum(psd, dim=0)
        total_energy = cumsum_psd[-1:, :]
        rolloff_threshold = 0.85 * total_energy
        spectral_rolloff = torch.argmax((cumsum_psd >= rolloff_threshold).float(), dim=0).float()
        
        # Spectral flatness (measure of noisiness)
        geometric_mean = torch.exp(torch.mean(torch.log(psd + 1e-8), dim=0))
        arithmetic_mean = torch.mean(psd, dim=0)
        spectral_flatness = geometric_mean / (arithmetic_mean + 1e-8)
        
        # High-frequency energy ratio
        mid_point = self.num_frequencies // 2
        low_energy = torch.sum(psd[:mid_point, :], dim=0)
        high_energy = torch.sum(psd[mid_point:, :], dim=0)
        hf_ratio = high_energy / (low_energy + high_energy + 1e-8)
        
        # Frequency domain uncertainty features
        freq_uncertainty = torch.std(torch.abs(fft_preds), dim=0)
        phase_uncertainty = torch.std(torch.angle(fft_preds), dim=0)
        
        # Stack all frequency features
        frequency_features = torch.stack([
            spectral_centroid, spectral_rolloff, spectral_flatness, 
            hf_ratio, freq_uncertainty, phase_uncertainty
        ], dim=1)  # [batch_size, 6]
        
        # Use subset for frequency analyzer (pad or truncate to num_frequencies)
        if frequency_features.shape[1] < self.num_frequencies:
            padding = torch.zeros(frequency_features.shape[0], 
                                self.num_frequencies - frequency_features.shape[1], 
                                device=frequency_features.device)
            frequency_features = torch.cat([frequency_features, padding], dim=1)
        else:
            frequency_features = frequency_features[:, :self.num_frequencies]
        
        return frequency_features, psd
    
    def predict_with_uncertainty(self, clinical_features, patient_embeddings, num_samples=100):
        """Get predictions with frequency-based uncertainty analysis"""
        self.train()  # Enable dropout during inference
        
        predictions = []
        for _ in range(num_samples):
            with torch.no_grad():
                if hasattr(self.base_model, 'forward'):
                    pred = self.base_model(clinical_features, patient_embeddings)
                    # Handle different output formats
                    if isinstance(pred, tuple):
                        pred = pred[0]  # Take first element if tuple
                    pred = torch.sigmoid(pred)
                    predictions.append(pred)
        
        # Stack predictions: [num_samples, batch_size, 1]
        predictions_tensor = torch.stack(predictions, dim=0)
        
        # Compute frequency features
        frequency_features, psd = self._compute_frequency_features(predictions_tensor)
        
        # Analyze frequency features for uncertainty
        freq_uncertainty_metrics = self.frequency_analyzer(frequency_features)
        
        # Traditional MC statistics
        mean_pred = torch.mean(predictions_tensor, dim=0)
        std_uncertainty = torch.std(predictions_tensor, dim=0)
        
        # Advanced frequency-based uncertainties
        frequency_uncertainty = freq_uncertainty_metrics[:, 0:1]
        spectral_density = freq_uncertainty_metrics[:, 1:2]
        coherence = torch.sigmoid(freq_uncertainty_metrics[:, 2:3])  # Coherence in [0,1]
        
        # Composite uncertainty combining traditional and frequency-based
        composite_uncertainty = (
            0.4 * std_uncertainty + 
            0.3 * frequency_uncertainty + 
            0.3 * (1 - coherence)  # Low coherence = high uncertainty
        )
        
        return mean_pred.cpu().numpy(), {
            'traditional_uncertainty': std_uncertainty.cpu().numpy(),
            'frequency_uncertainty': frequency_uncertainty.cpu().numpy(),
            'spectral_density': spectral_density.cpu().numpy(),
            'coherence': coherence.cpu().numpy(),
            'composite_uncertainty': composite_uncertainty.cpu().numpy(),
            'power_spectral_density': psd.cpu().numpy()
        }

# =============================================================================
# Uncertainty Estimation Methods
# =============================================================================

class MCDropoutModel(nn.Module):
    """Monte Carlo Dropout for uncertainty estimation"""
    
    def __init__(self, base_model, dropout_rate=0.1):
        super().__init__()
        self.base_model = base_model
        self.dropout_rate = dropout_rate
        self.clinical_dim = base_model.clinical_dim
        self.embedding_dim = base_model.embedding_dim
        
        # Replace all dropout layers to be active during inference
        self._replace_dropouts(self.base_model)
    
    def _replace_dropouts(self, module):
        """Replace standard dropout with MC dropout"""
        for name, child in module.named_children():
            if isinstance(child, nn.Dropout):
                setattr(module, name, nn.Dropout(p=self.dropout_rate))
            else:
                self._replace_dropouts(child)
    
    def forward(self, clinical_features, patient_embeddings):
        return self.base_model(clinical_features, patient_embeddings)
    
    def predict_with_uncertainty(self, clinical_features, patient_embeddings, num_samples=100):
        """Get predictions with uncertainty estimates"""
        self.train()  # Enable dropout during inference
        
        predictions = []
        for _ in range(num_samples):
            with torch.no_grad():
                pred = self.forward(clinical_features, patient_embeddings)
                predictions.append(torch.sigmoid(pred).cpu().numpy())
        
        predictions = np.array(predictions)
        mean_pred = np.mean(predictions, axis=0)
        uncertainty = np.std(predictions, axis=0)
        
        return mean_pred, uncertainty

class VariationalModel(BaseMultimodalModel):
    """Variational Bayesian Neural Network with advanced clinical processing"""
    
    def __init__(self, clinical_dim, embedding_dim, hidden_dim=256, dropout=0.1, 
                 feature_schema=None, feature_order=None):
        super().__init__(clinical_dim, embedding_dim, hidden_dim, dropout)
        
        # Variational clinical encoders
        if feature_schema and feature_order:
            self.clinical_encoder_mean = DomainAwareClinicalEncoder(
                clinical_dim, embedding_dim, feature_schema, feature_order, hidden_dim, dropout
            )
            self.clinical_encoder_logvar = DomainAwareClinicalEncoder(
                clinical_dim, embedding_dim, feature_schema, feature_order, hidden_dim, dropout
            )
        else:
            self.clinical_encoder_mean = SimpleClinicalEncoder(clinical_dim, embedding_dim, hidden_dim, dropout)
            self.clinical_encoder_logvar = SimpleClinicalEncoder(clinical_dim, embedding_dim, hidden_dim, dropout)
        
        self.classifier_mean = nn.Linear(embedding_dim * 2, 1)
        self.classifier_logvar = nn.Linear(embedding_dim * 2, 1)
    
    def reparameterize(self, mu, logvar):
        """Reparameterization trick"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, clinical_features, patient_embeddings):
        # Variational clinical encoding
        clinical_mu = self.clinical_encoder_mean(clinical_features)
        clinical_logvar = self.clinical_encoder_logvar(clinical_features)
        clinical_encoded = self.reparameterize(clinical_mu, clinical_logvar)
        
        # Combine features
        combined = torch.cat([clinical_encoded, patient_embeddings], dim=1)
        
        # Variational prediction
        pred_mu = self.classifier_mean(combined)
        pred_logvar = self.classifier_logvar(combined)
        
        # KL divergence loss
        kl_loss = -0.5 * torch.sum(1 + clinical_logvar - clinical_mu.pow(2) - clinical_logvar.exp())
        
        return pred_mu, pred_logvar, kl_loss

# =============================================================================
# Novel Uncertainty Method: Multimodal Uncertainty Decomposition (MUD)
# =============================================================================

class MultimodalUncertaintyDecomposition(BaseMultimodalModel):
    """Novel: Decompose uncertainty into modality-specific components with advanced clinical processing"""
    
    def __init__(self, clinical_dim, embedding_dim, hidden_dim=256, dropout=0.1, 
                 feature_schema=None, feature_order=None):
        super().__init__(clinical_dim, embedding_dim, hidden_dim, dropout)
        
        # Advanced clinical processing
        if feature_schema and feature_order:
            self.clinical_processor = DomainAwareClinicalEncoder(
                clinical_dim, hidden_dim, feature_schema, feature_order, hidden_dim, dropout
            )
            clinical_pred_input = hidden_dim
        else:
            self.clinical_processor = None
            clinical_pred_input = clinical_dim
        
        # Separate uncertainty predictors for each modality
        self.clinical_predictor = nn.Sequential(
            nn.Linear(clinical_pred_input, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2)  # [prediction, uncertainty]
        )
        
        self.imaging_predictor = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2)  # [prediction, uncertainty]
        )
        
        # Fusion uncertainty predictor
        self.fusion_uncertainty = nn.Sequential(
            nn.Linear(4, hidden_dim // 2),  # 2 preds + 2 uncertainties
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)  # Additional fusion uncertainty
        )
        
        # Cross-modal confidence predictor
        cross_modal_input = hidden_dim + embedding_dim if self.clinical_processor else clinical_dim + embedding_dim
        self.cross_modal_confidence = nn.Sequential(
            nn.Linear(cross_modal_input, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2)  # [clinical_trust_imaging, imaging_trust_clinical]
        )
    
    def forward(self, clinical_features, patient_embeddings):
        # Process clinical features if advanced encoder available
        if self.clinical_processor:
            clinical_processed = self.clinical_processor(clinical_features)
            cross_modal_input = torch.cat([clinical_processed, patient_embeddings], dim=1)
        else:
            clinical_processed = clinical_features
            cross_modal_input = torch.cat([clinical_features, patient_embeddings], dim=1)
        
        # Individual modality predictions and uncertainties
        clinical_out = self.clinical_predictor(clinical_processed)
        imaging_out = self.imaging_predictor(patient_embeddings)
        
        clinical_pred, clinical_unc = clinical_out[:, 0:1], torch.exp(clinical_out[:, 1:2])
        imaging_pred, imaging_unc = imaging_out[:, 0:1], torch.exp(imaging_out[:, 1:2])
        
        # Fusion uncertainty
        fusion_input = torch.cat([clinical_pred, imaging_pred, clinical_unc, imaging_unc], dim=1)
        fusion_unc = torch.exp(self.fusion_uncertainty(fusion_input))
        
        # Cross-modal confidence
        cross_confidence = torch.sigmoid(self.cross_modal_confidence(cross_modal_input))
        clinical_trust_imaging, imaging_trust_clinical = cross_confidence[:, 0:1], cross_confidence[:, 1:2]
        
        # Uncertainty-weighted fusion
        clinical_weight = 1.0 / (clinical_unc + 1e-8) * imaging_trust_clinical
        imaging_weight = 1.0 / (imaging_unc + 1e-8) * clinical_trust_imaging
        
        total_weight = clinical_weight + imaging_weight + 1e-8
        final_pred = (clinical_pred * clinical_weight + imaging_pred * imaging_weight) / total_weight
        
        # Total uncertainty (combining all sources)
        total_uncertainty = clinical_unc + imaging_unc + fusion_unc
        
        return final_pred, {
            'clinical_uncertainty': clinical_unc,
            'imaging_uncertainty': imaging_unc,
            'fusion_uncertainty': fusion_unc,
            'total_uncertainty': total_uncertainty,
            'clinical_trust_imaging': clinical_trust_imaging,
            'imaging_trust_clinical': imaging_trust_clinical
        }

# =============================================================================
# Ensemble Methods
# =============================================================================

class DeepEnsemble:
    """Deep ensemble for uncertainty estimation"""
    
    def __init__(self, model_class, num_models=5, **model_kwargs):
        self.models = []
        self.num_models = num_models
        for _ in range(num_models):
            model = model_class(**model_kwargs)
            self.models.append(model)
    
    def train_ensemble(self, train_loader, val_loader, epochs=50, device='cuda'):
        """Train all models in ensemble"""
        for i, model in enumerate(self.models):
            print(f"Training ensemble model {i+1}/{len(self.models)}")
            # Training code would go here - implement as needed
            pass
    
    def predict_with_uncertainty(self, clinical_features, patient_embeddings):
        """Get ensemble predictions with uncertainty"""
        predictions = []
        
        for model in self.models:
            model.eval()
            with torch.no_grad():
                pred = torch.sigmoid(model(clinical_features, patient_embeddings))
                predictions.append(pred.cpu().numpy())
        
        predictions = np.array(predictions)
        mean_pred = np.mean(predictions, axis=0)
        uncertainty = np.std(predictions, axis=0)
        
        return mean_pred, uncertainty

# =============================================================================
# Model Factory
# =============================================================================

def create_model(model_type, clinical_dim, embedding_dim, feature_schema=None, feature_order=None, **kwargs):
    """Factory function to create models with advanced clinical processing"""
    
    # Add feature schema and order to kwargs if provided
    if feature_schema and feature_order:
        kwargs['feature_schema'] = feature_schema
        kwargs['feature_order'] = feature_order
    
    models = {
        # Non-multimodal baselines
        'clinical_only': ClinicalOnlyModel,
        'imaging_only': ImagingOnlyModel,
        
        # Simple fusion methods
        'simple_concat': SimpleConcatenationModel,
        'early_fusion': EarlyFusionModel,
        'late_fusion': LateFusionModel,
        
        # Advanced fusion methods
        'contrastive': ContrastiveFusionModel,
        'cross_attention': CrossAttentionFusionModel,
        'bilinear': BilinearFusionModel,
        
        # Novel fusion methods
        'ccg': ClinicalContextualGating,
        
        # Uncertainty estimation methods
        'variational': VariationalModel,
        'mud': MultimodalUncertaintyDecomposition,
        'evidential': EvidentialModel,
    }
    
    if model_type not in models:
        raise ValueError(f"Unknown model type: {model_type}. Available: {list(models.keys())}")
    
    base_model = models[model_type](clinical_dim, embedding_dim, **kwargs)
    
    # Handle special uncertainty wrapper models
    if model_type == 'mc_frequency':
        # Create base model and wrap with MC frequency analyzer
        base_model_type = kwargs.get('base_model_type', 'simple_concat')
        if base_model_type in models:
            base_model = models[base_model_type](clinical_dim, embedding_dim, **kwargs)
            return MonteCarloFrequencyModel(base_model, kwargs.get('dropout', 0.1))
        else:
            raise ValueError(f"Base model type {base_model_type} not found for MC frequency analysis")
    
    return base_model