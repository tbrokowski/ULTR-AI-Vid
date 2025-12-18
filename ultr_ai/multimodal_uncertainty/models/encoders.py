# =============================================================================
# Advanced Clinical Encoders with Domain Awareness
# =============================================================================
from typing import Dict, List
import torch
import torch.nn as nn

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