# =============================================================================
# Advanced Uncertainty Estimation Methods
# =============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from ultr_ai.multimodal_uncertainty.base import BaseMultimodalModel
from ultr_ai.multimodal_uncertainty.encoders import DomainAwareClinicalEncoder, SimpleClinicalEncoder



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