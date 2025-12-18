# =============================================================================
# Ensemble Methods
# =============================================================================

import torch
import numpy as np

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