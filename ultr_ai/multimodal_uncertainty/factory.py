# =============================================================================
# Model Factory
# =============================================================================

from ultr_ai.multimodal_uncertainty.models import *

def create_model(model_type, clinical_dim, embedding_dim, feature_schema=None, feature_order=None, **kwargs):
    """Factory function to create models with advanced clinical processing"""
    
    # Add feature schema and order to kwargs if provided
    if feature_schema and feature_order:
        kwargs['feature_schema'] = feature_schema
        kwargs['feature_order'] = feature_order
    
    # Special-case wrapper models that aren't direct classifiers
    # Handle MC Frequency before validating against the base models map
    if model_type == 'mc_frequency':
        # Determine the underlying base model type (default to simple_concat)
        base_model_type = kwargs.pop('base_model_type', 'simple_concat')
        # Build the base model using the standard registry below
        base_models = {
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
            # Uncertainty estimation base models
            'variational': VariationalModel,
            'mud': MultimodalUncertaintyDecomposition,
            'evidential': EvidentialModel,
        }
        if base_model_type not in base_models:
            raise ValueError(f"Base model type {base_model_type} not found for MC frequency analysis")
        base_model = base_models[base_model_type](clinical_dim, embedding_dim, **kwargs)
        # Wrap with MC frequency analyzer
        return MonteCarloFrequencyModel(base_model, kwargs.get('dropout', 0.1))

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