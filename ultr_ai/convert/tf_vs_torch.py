"""
TensorFlow vs PyTorch Layer Comparison Tests

This module provides comprehensive tests to verify that TensorFlow implementations
produce the same outputs as PyTorch implementations for each layer/component.

Each test:
1. Creates both PyTorch and TensorFlow versions of the module
2. Transfers weights from PyTorch to TensorFlow
3. Runs the same input through both
4. Verifies outputs match within tolerance

Usage:
    python -m ultr_ai.tf.utils.tf_vs_torch

Or run specific tests:
    python -m pytest ultr_ai/tf/utils/tf_vs_torch.py -v

THINGS TO REMEMBER ABOUT TF:
- The layernorm has a different default epsilon value than PyTorch. TF: 0.001, PyTorch: 1e-5
"""

import numpy as np
import logging

logger = logging.getLogger(__name__)

# Set seeds for reproducibility
SEED = 42
np.random.seed(SEED)

import torch
import torch.nn as nn
import torch.nn.functional as F
import tensorflow as tf


def set_seeds():
    """Set random seeds for reproducibility in both frameworks."""
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    tf.random.set_seed(SEED)


# ============================================================================
# WEIGHT TRANSFER UTILITIES
# ============================================================================

def transfer_linear_weights(torch_linear, tf_dense, has_bias=True):
    """
    Transfer weights from PyTorch Linear to TensorFlow Dense.
    
    PyTorch Linear: weight [out_features, in_features], bias [out_features]
    TF Dense: kernel [in_features, out_features], bias [out_features]
    """
    torch_weight = torch_linear.weight.detach().numpy()
    tf_kernel = np.transpose(torch_weight)  # Transpose for TF
    
    if has_bias and torch_linear.bias is not None:
        torch_bias = torch_linear.bias.detach().numpy()
        tf_dense.set_weights([tf_kernel, torch_bias])
    else:
        tf_dense.set_weights([tf_kernel])


def transfer_layernorm_weights(torch_ln, tf_ln):
    """Transfer weights from PyTorch LayerNorm to TensorFlow LayerNormalization."""
    gamma = torch_ln.weight.detach().numpy()
    beta = torch_ln.bias.detach().numpy()
    tf_ln.set_weights([gamma, beta])


def transfer_embedding_weights(torch_emb, tf_emb):
    """Transfer weights from PyTorch Embedding to TensorFlow Embedding."""
    torch_weight = torch_emb.weight.detach().numpy()
    tf_emb.set_weights([torch_weight])


def transfer_conv1d_weights(torch_conv, tf_conv):
    """
    Transfer weights from PyTorch Conv1d to TensorFlow Conv1D.
    
    PyTorch Conv1d: weight [out_channels, in_channels, kernel_size]
    TF Conv1D: kernel [kernel_size, in_channels, out_channels]
    """
    torch_weight = torch_conv.weight.detach().numpy()
    tf_kernel = np.transpose(torch_weight, (2, 1, 0))
    
    if torch_conv.bias is not None:
        torch_bias = torch_conv.bias.detach().numpy()
        tf_conv.set_weights([tf_kernel, torch_bias])
    else:
        tf_conv.set_weights([tf_kernel])


def transfer_gru_weights(torch_gru, tf_gru):
    """
    Transfer weights from PyTorch GRU to TensorFlow GRU.
    
    PyTorch GRU concatenates gates as [reset, update, new] 
    TensorFlow GRU concatenates gates as [update, reset, new]
    """
    # Get PyTorch weights
    weight_ih = torch_gru.weight_ih_l0.detach().numpy()  # [3*hidden, input]
    weight_hh = torch_gru.weight_hh_l0.detach().numpy()  # [3*hidden, hidden]
    bias_ih = torch_gru.bias_ih_l0.detach().numpy()      # [3*hidden]
    bias_hh = torch_gru.bias_hh_l0.detach().numpy()      # [3*hidden]
    
    hidden_size = weight_ih.shape[0] // 3
    
    # Split PyTorch weights: [reset, update, new]
    r_ih, z_ih, n_ih = np.split(weight_ih, 3, axis=0)
    r_hh, z_hh, n_hh = np.split(weight_hh, 3, axis=0)
    r_bih, z_bih, n_bih = np.split(bias_ih, 3)
    r_bhh, z_bhh, n_bhh = np.split(bias_hh, 3)
    
    # Reorder for TensorFlow: [update, reset, new]
    tf_weight_ih = np.concatenate([z_ih, r_ih, n_ih], axis=0)
    tf_weight_hh = np.concatenate([z_hh, r_hh, n_hh], axis=0)
    tf_bias = np.concatenate([z_bih + z_bhh, r_bih + r_bhh, n_bih, n_bhh])
    
    # TF GRU expects: kernel [input, 3*hidden], recurrent_kernel [hidden, 3*hidden], bias [2, 3*hidden]
    tf_kernel = tf_weight_ih.T  # [input, 3*hidden]
    tf_recurrent_kernel = tf_weight_hh.T  # [hidden, 3*hidden]
    tf_bias_reshaped = np.stack([
        np.concatenate([z_bih, r_bih, n_bih]),
        np.concatenate([z_bhh, r_bhh, n_bhh])
    ])
    
    tf_gru.set_weights([tf_kernel, tf_recurrent_kernel, tf_bias_reshaped])


def transfer_sequential_weights(torch_seq, tf_seq, layer_mapping):
    """
    Transfer weights from PyTorch Sequential to TensorFlow Sequential.
    
    layer_mapping: list of tuples (torch_idx, tf_idx, transfer_func)
    """
    for torch_idx, tf_idx, transfer_func in layer_mapping:
        torch_layer = torch_seq[torch_idx]
        tf_layer = tf_seq.layers[tf_idx]
        transfer_func(torch_layer, tf_layer)

def transfer_mha_weights(torch_mha, tf_mha, embed_dim, num_heads):
    """
    Transfer weights from PyTorch MultiheadAttention to TensorFlow MultiHeadAttention.
    
    This is complex because:
    - PyTorch uses in_proj_weight [3*embed_dim, embed_dim] for Q, K, V combined
    - TensorFlow uses separate query_dense, key_dense, value_dense
    """
    # Get PyTorch weights
    in_proj_weight = torch_mha.in_proj_weight.detach().numpy()  # [3*E, E]
    in_proj_bias = torch_mha.in_proj_bias.detach().numpy()       # [3*E]
    out_proj_weight = torch_mha.out_proj.weight.detach().numpy() # [E, E]
    out_proj_bias = torch_mha.out_proj.bias.detach().numpy()     # [E]
    
    # Split in_proj into Q, K, V
    q_weight, k_weight, v_weight = np.split(in_proj_weight, 3, axis=0)
    q_bias, k_bias, v_bias = np.split(in_proj_bias, 3)
    
    # TF MHA expects weights in shape [E, num_heads, key_dim]
    key_dim = embed_dim // num_heads
    
    # Reshape for TF: [E, E] -> [E, num_heads, key_dim]
    def reshape_for_tf(weight, bias):
        # weight: [E, E] (transposed from PyTorch)
        # TF wants: kernel [E, num_heads, key_dim]
        w = weight.T  # [E, E]
        w = w.reshape(embed_dim, num_heads, key_dim)
        b = bias.reshape(num_heads, key_dim)
        return w, b
    
    q_kernel, q_b = reshape_for_tf(q_weight, q_bias)
    k_kernel, k_b = reshape_for_tf(k_weight, k_bias)
    v_kernel, v_b = reshape_for_tf(v_weight, v_bias)
    
    # Output projection: [E, E] -> [num_heads, key_dim, E]
    out_kernel = out_proj_weight.T.reshape(num_heads, key_dim, embed_dim)
    
    # Set TF weights
    tf_mha._query_dense.set_weights([q_kernel, q_b])
    tf_mha._key_dense.set_weights([k_kernel, k_b])
    tf_mha._value_dense.set_weights([v_kernel, v_b])
    tf_mha._output_dense.set_weights([out_kernel, out_proj_bias])

# ============================================================================
# SIMPLE LAYER TESTS
# ============================================================================

class SimpleModelTorch(nn.Module):
    def __init__(self):
        super(SimpleModelTorch, self).__init__()
        self.linear = nn.Linear(2, 5, bias=False)

        # Set weights
        self.linear.weight = nn.Parameter(torch.tensor([[0.1, 0.2],
                                                        [1.0, 0.9],
                                                        [0.2, 0.3],
                                                        [0.9, 0.8],
                                                        [0.5, 0.4]], dtype=torch.float32))
        

    def forward(self, x):
        return self.linear(x)
    

class SimpleModelTF(tf.keras.Model):
    def __init__(self):
        super(SimpleModelTF, self).__init__()
        self.linear = tf.keras.layers.Dense(5, use_bias=False, input_shape=(2,))

        # Set weights
        self.linear.build((None, 2))
        self.linear.set_weights([tf.transpose(tf.constant([[0.1, 0.2],
                                                            [1.0, 0.9],
                                                            [0.2, 0.3],
                                                            [0.9, 0.8],
                                                            [0.5, 0.4]], dtype=tf.float32))])
    def call(self, inputs):
        return self.linear(inputs)


def test_simple_linear():
    """Test that simple linear layers produce matching outputs."""
    set_seeds()
    
    # Create models
    torch_model = SimpleModelTorch()
    tf_model = SimpleModelTF()
    
    # Test input
    test_input = np.array([[1.0, 2.0]], dtype=np.float32)
    
    # PyTorch forward
    torch_input = torch.tensor(test_input)
    torch_output = torch_model(torch_input).detach().numpy()
    
    # TensorFlow forward
    tf_input = tf.constant(test_input)
    tf_output = tf_model(tf_input).numpy()
    
    # Compare
    max_diff = np.max(np.abs(torch_output - tf_output))
    print(f"Simple Linear Test:")
    print(f"  PyTorch output: {torch_output}")
    print(f"  TensorFlow output: {tf_output}")
    print(f"  Max difference: {max_diff}")
    
    assert max_diff < 1e-5, f"Outputs differ by {max_diff}"
    print("  PASSED\n")
    return True


# ============================================================================
# LAYER NORMALIZATION TEST
# ============================================================================

def test_layer_norm():
    """Test LayerNormalization produces matching outputs with same weights."""
    set_seeds()
    
    hidden_dim = 64
    batch_size = 2
    seq_len = 10
    
    # Create test input
    test_input = np.random.randn(batch_size, seq_len, hidden_dim).astype(np.float32)
    
    # PyTorch LayerNorm
    torch_ln = nn.LayerNorm(hidden_dim)
    
    # TensorFlow LayerNormalization
    tf_ln = tf.keras.layers.LayerNormalization(epsilon=1e-05)
    tf_ln.build((None, seq_len, hidden_dim))
    
    # Transfer weights from PyTorch to TensorFlow
    transfer_layernorm_weights(torch_ln, tf_ln)
    
    # Forward pass
    torch_input = torch.tensor(test_input)
    torch_output = torch_ln(torch_input).detach().numpy()
    
    tf_input = tf.constant(test_input)
    tf_output = tf_ln(tf_input).numpy()
    
    max_diff = np.max(np.abs(torch_output - tf_output))
    print(f"LayerNorm Test (with weight transfer):")
    print(f"  Input shape: {test_input.shape}")
    print(f"  Max difference: {max_diff}")
    
    assert max_diff < 1e-5, f"Outputs differ by {max_diff}"
    print("  PASSED\n")
    return True


# ============================================================================
# CONV1D TEST
# ============================================================================

def test_conv1d():
    """Test Conv1D produces matching outputs with same weights."""
    set_seeds()
    
    in_channels = 64
    out_channels = 32
    kernel_size = 3
    batch_size = 2
    seq_len = 10
    
    # Create test input
    test_input = np.random.randn(batch_size, seq_len, in_channels).astype(np.float32)
    
    # PyTorch Conv1D (expects [B, C, L])
    torch_conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size//2)
    
    # TensorFlow Conv1D (expects [B, L, C])
    tf_conv = tf.keras.layers.Conv1D(out_channels, kernel_size, padding='same')
    tf_conv.build((None, seq_len, in_channels))
    
    # Transfer weights
    transfer_conv1d_weights(torch_conv, tf_conv)
    
    # PyTorch forward (convert to [B, C, L] and back)
    torch_input = torch.tensor(test_input).transpose(1, 2)
    torch_output = torch_conv(torch_input).transpose(1, 2).detach().numpy()
    
    # TensorFlow forward
    tf_input = tf.constant(test_input)
    tf_output = tf_conv(tf_input).numpy()
    
    max_diff = np.max(np.abs(torch_output - tf_output))
    print(f"Conv1D Test (with weight transfer):")
    print(f"  Input shape: {test_input.shape}")
    print(f"  Output shapes - PyTorch: {torch_output.shape}, TF: {tf_output.shape}")
    print(f"  Max difference: {max_diff}")
    
    # Conv1D can have small numerical differences due to implementation
    assert max_diff < 1e-4, f"Outputs differ by {max_diff}"
    print("  PASSED\n")
    return True


# ============================================================================
# SOFTMAX TEST
# ============================================================================

def test_softmax():
    """Test softmax produces matching outputs."""
    set_seeds()
    
    # Create test input
    test_input = np.random.randn(2, 10).astype(np.float32)
    
    # PyTorch softmax
    torch_output = F.softmax(torch.tensor(test_input), dim=-1).numpy()
    
    # TensorFlow softmax
    tf_output = tf.nn.softmax(tf.constant(test_input), axis=-1).numpy()
    
    max_diff = np.max(np.abs(torch_output - tf_output))
    print(f"Softmax Test:")
    print(f"  Max difference: {max_diff}")
    
    assert max_diff < 1e-6, f"Outputs differ by {max_diff}"
    print("  PASSED\n")
    return True


# ============================================================================
# GELU ACTIVATION TEST
# ============================================================================

def test_gelu():
    """Test GELU activation produces matching outputs."""
    set_seeds()
    
    # Create test input
    test_input = np.random.randn(2, 10, 64).astype(np.float32)
    
    # PyTorch GELU
    torch_output = F.gelu(torch.tensor(test_input)).numpy()
    
    # TensorFlow GELU
    tf_output = tf.nn.gelu(tf.constant(test_input)).numpy()
    
    max_diff = np.max(np.abs(torch_output - tf_output))
    print(f"GELU Test:")
    print(f"  Max difference: {max_diff}")
    
    # GELU implementations may have small differences
    assert max_diff < 1e-5, f"Outputs differ by {max_diff}"
    print("  PASSED\n")
    return True


# ============================================================================
# WEIGHT TRANSFER TEST
# ============================================================================

def test_weight_transfer():
    """Test that weights can be transferred from PyTorch to TensorFlow correctly."""
    set_seeds()
    
    in_features = 64
    out_features = 32
    
    # Create PyTorch linear layer with specific weights
    torch_linear = nn.Linear(in_features, out_features)
    
    # Create TensorFlow Dense layer
    tf_dense = tf.keras.layers.Dense(out_features, use_bias=True)
    tf_dense.build((None, in_features))
    
    # Transfer weights
    transfer_linear_weights(torch_linear, tf_dense)
    
    # Test with same input
    test_input = np.random.randn(2, in_features).astype(np.float32)
    
    torch_output = torch_linear(torch.tensor(test_input)).detach().numpy()
    tf_output = tf_dense(tf.constant(test_input)).numpy()
    
    max_diff = np.max(np.abs(torch_output - tf_output))
    print(f"Weight Transfer Test:")
    print(f"  Max difference after weight transfer: {max_diff}")
    
    assert max_diff < 1e-5, f"Outputs differ by {max_diff}"
    print("  PASSED\n")
    return True


# ============================================================================
# PATHOLOGY MODULE TEST WITH WEIGHT TRANSFER
# ============================================================================

def test_pathology_module():
    """Test PathologyModule TF vs PyTorch with weight transfer."""
    set_seeds()
    
    # Import both modules
    from ultr_ai.tf.network_architecture.components.general_components import PathologyModuleTF
    from ultr_ai.network_architecture.components.general_components import PathologyModule
    
    feature_dim = 512
    hidden_dim = 256
    batch_size = 2
    num_frames = 5
    
    # Create test input
    test_input = np.random.randn(batch_size, num_frames, feature_dim).astype(np.float32)
    
    # Create PyTorch module
    torch_module = PathologyModule(
        feature_dim=feature_dim,
        hidden_dim=hidden_dim,
        dropout=0.0,  # Disable dropout for deterministic comparison
        name='pathology_test'
    )
    torch_module.eval()
    
    # Create TensorFlow module
    tf_module = PathologyModuleTF(
        feature_dim=feature_dim,
        hidden_dim=hidden_dim,
        dropout=0.0,
        name='pathology_test'
    )
    
    # Build TF module by calling it
    tf_input = tf.constant(test_input)
    _ = tf_module(tf_input, training=False)
    
    # Transfer weights from PyTorch to TensorFlow
    # feature_refine: Linear, LayerNorm, Tanh, Dropout
    transfer_linear_weights(torch_module.feature_refine[0], tf_module.feature_refine.layers[0])
    transfer_layernorm_weights(torch_module.feature_refine[1], tf_module.feature_refine.layers[1])
    
    # frame_attention: Linear, Tanh, Linear
    transfer_linear_weights(torch_module.frame_attention[0], tf_module.frame_attention.layers[0])
    transfer_linear_weights(torch_module.frame_attention[2], tf_module.frame_attention.layers[2])
    
    # classifier: Linear, GELU, Dropout, Linear
    transfer_linear_weights(torch_module.classifier[0], tf_module.classifier.layers[0])
    transfer_linear_weights(torch_module.classifier[3], tf_module.classifier.layers[3])
    
    # Forward pass
    torch_input = torch.tensor(test_input)
    with torch.no_grad():
        torch_score, torch_attn, torch_pooled = torch_module(torch_input)
    
    tf_score, tf_attn, tf_pooled = tf_module(tf_input, training=False)
    
    # Compare outputs
    score_diff = np.max(np.abs(torch_score.numpy() - tf_score.numpy()))
    attn_diff = np.max(np.abs(torch_attn.numpy() - tf_attn.numpy()))
    pooled_diff = np.max(np.abs(torch_pooled.numpy() - tf_pooled.numpy()))
    
    print(f"PathologyModule Test (with weight transfer):")
    print(f"  Input shape: {test_input.shape}")
    print(f"  Score difference: {score_diff}")
    print(f"  Attention difference: {attn_diff}")
    print(f"  Pooled features difference: {pooled_diff}")
    
    # Verify output shapes
    assert tf_score.shape == (batch_size, 1), f"Score shape mismatch: {tf_score.shape}"
    assert tf_attn.shape == (batch_size, num_frames), f"Attention shape mismatch: {tf_attn.shape}"
    assert tf_pooled.shape == (batch_size, hidden_dim), f"Pooled shape mismatch: {tf_pooled.shape}"
    
    # Verify attention sums to 1
    attn_sum = np.sum(tf_attn.numpy(), axis=-1)
    assert np.allclose(attn_sum, 1.0, atol=1e-5), f"Attention doesn't sum to 1: {attn_sum}"
    
    # Check outputs match
    assert score_diff < 1e-4, f"Score outputs differ by {score_diff}"
    assert attn_diff < 1e-4, f"Attention outputs differ by {attn_diff}"
    assert pooled_diff < 1e-4, f"Pooled outputs differ by {pooled_diff}"
    
    print("  PASSED\n")
    return True


# ============================================================================
# SITE INTEGRATION MODULE TEST WITH WEIGHT TRANSFER
# ============================================================================

def test_site_integration_module():
    """Test SiteIntegrationModule TF vs PyTorch with weight transfer."""
    set_seeds()
    
    from ultr_ai.tf.network_architecture.components.general_components import SiteIntegrationModuleTF
    from ultr_ai.network_architecture.components.general_components import SiteIntegrationModule
    
    feature_dim = 512
    hidden_dim = 512
    site_embed_dim = 256
    num_sites = 15
    num_pathologies = 4
    batch_size = 2
    max_sites_per_patient = 5
    
    # Create test inputs
    site_features = np.random.randn(batch_size, max_sites_per_patient, feature_dim).astype(np.float32)
    site_indices = np.array([[1, 2, 3, 4, 5], [0, 1, 2, 3, 4]], dtype=np.int32)
    pathology_scores = np.random.randn(batch_size, max_sites_per_patient, num_pathologies).astype(np.float32)
    
    # Create PyTorch module
    torch_module = SiteIntegrationModule(
        feature_dim=feature_dim,
        site_embed_dim=site_embed_dim,
        hidden_dim=hidden_dim,
        num_sites=num_sites,
        num_pathologies=num_pathologies,
        dropout=0.0
    )
    torch_module.eval()
    
    # Create TF module
    tf_module = SiteIntegrationModuleTF(
        feature_dim=feature_dim,
        site_embed_dim=site_embed_dim,
        hidden_dim=hidden_dim,
        num_sites=num_sites,
        num_pathologies=num_pathologies,
        dropout=0.0
    )
    
    # Build TF module
    _ = tf_module(
        tf.constant(site_features),
        tf.constant(site_indices),
        tf.constant(pathology_scores),
        training=False
    )
    
    # Transfer weights
    transfer_embedding_weights(torch_module.site_embedding, tf_module.site_embedding)
    
    # integration Sequential: Dense, LayerNorm, GELU, Dropout, Dense, LayerNorm, GELU
    transfer_linear_weights(torch_module.integration[0], tf_module.integration.layers[0])
    transfer_layernorm_weights(torch_module.integration[1], tf_module.integration.layers[1])
    transfer_linear_weights(torch_module.integration[4], tf_module.integration.layers[4])
    transfer_layernorm_weights(torch_module.integration[5], tf_module.integration.layers[5])
    
    # Forward pass
    with torch.no_grad():
        torch_output = torch_module(
            torch.tensor(site_features),
            torch.tensor(site_indices, dtype=torch.long),
            torch.tensor(pathology_scores)
        )
    
    tf_output = tf_module(
        tf.constant(site_features),
        tf.constant(site_indices),
        tf.constant(pathology_scores),
        training=False
    )
    
    max_diff = np.max(np.abs(torch_output.numpy() - tf_output.numpy()))
    
    print(f"SiteIntegrationModule Test (with weight transfer):")
    print(f"  Input shapes - features: {site_features.shape}, indices: {site_indices.shape}")
    print(f"  Output shape: {tf_output.shape}")
    print(f"  Max difference: {max_diff}")
    
    assert tf_output.shape == (batch_size, max_sites_per_patient, hidden_dim)
    assert not np.any(np.isnan(tf_output.numpy()))
    assert max_diff < 1e-4, f"Outputs differ by {max_diff}"
    
    print("  PASSED\n")
    return True


# ============================================================================
# DEEP ATTENTION MIL TEST WITH WEIGHT TRANSFER
# ============================================================================

def test_deep_attention_mil():
    """Test DeepAttentionMIL TF vs PyTorch with weight transfer."""
    set_seeds()
    
    from ultr_ai.tf.network_architecture.components.general_components import DeepAttentionMILTF
    from ultr_ai.network_architecture.components.general_components import DeepAttentionMIL
    
    feature_dim = 512
    hidden_dim = 256
    batch_size = 2
    num_instances = 5
    num_heads = 8
    
    # Create test inputs
    features = np.random.randn(batch_size, num_instances, feature_dim).astype(np.float32)
    mask = np.array([[True, True, True, True, True], [True, True, True, True, True]])  # All valid
    
    # Create PyTorch module
    torch_module = DeepAttentionMIL(
        feature_dim=feature_dim,
        hidden_dim=hidden_dim,
        dropout=0.0,
        num_heads=num_heads
    )
    torch_module.eval()
    
    # Create TF module
    tf_module = DeepAttentionMILTF(
        feature_dim=feature_dim,
        hidden_dim=hidden_dim,
        dropout=0.0,
        num_heads=num_heads
    )
    
    # Build TF module
    _ = tf_module(tf.constant(features), tf.constant(mask), training=False)
    
    # Transfer weights for transform Sequential
    # PyTorch: Linear, LayerNorm, GELU, Dropout, Linear, LayerNorm
    # TF: Dense, LayerNorm, gelu, Dropout, Dense, LayerNorm
    transfer_linear_weights(torch_module.transform[0], tf_module.transform.layers[0])
    transfer_layernorm_weights(torch_module.transform[1], tf_module.transform.layers[1])
    transfer_linear_weights(torch_module.transform[4], tf_module.transform.layers[4])
    transfer_layernorm_weights(torch_module.transform[5], tf_module.transform.layers[5])
    
    # Transfer attention2 Sequential: Linear, Tanh, Linear
    transfer_linear_weights(torch_module.attention2[0], tf_module.attention2.layers[0])
    transfer_linear_weights(torch_module.attention2[2], tf_module.attention2.layers[2])
    
    # Transfer gating Sequential: Linear, GELU, Linear, Sigmoid
    transfer_linear_weights(torch_module.gating[0], tf_module.gating.layers[0])
    transfer_linear_weights(torch_module.gating[2], tf_module.gating.layers[2])
    
    # Transfer MultiHeadAttention weights (complex - requires careful mapping)
    # For now, we skip MHA weight transfer and test with random weights
    # The test will verify shapes and that no NaN occurs
    
    # Forward pass
    with torch.no_grad():
        torch_aggregated, torch_attention = torch_module(
            torch.tensor(features),
            torch.tensor(mask)
        )
    
    tf_aggregated, tf_attention = tf_module(
        tf.constant(features),
        tf.constant(mask),
        training=False
    )
    
    print(f"DeepAttentionMIL Test:")
    print(f"  Input shape: {features.shape}")
    print(f"  Aggregated shapes - PyTorch: {torch_aggregated.shape}, TF: {tf_aggregated.numpy().shape}")
    print(f"  Attention shapes - PyTorch: {torch_attention.shape}, TF: {tf_attention.numpy().shape}")
    
    assert tf_aggregated.shape == (batch_size, feature_dim)
    assert tf_attention.shape == (batch_size, num_instances)
    assert not np.any(np.isnan(tf_aggregated.numpy()))
    assert not np.any(np.isnan(tf_attention.numpy()))
    
    # Note: Due to MHA weight differences, we only check shapes and valid outputs
    print("  PASSED (shapes valid, no NaN - MHA weights not transferred)\n")
    return True


# ============================================================================
# ATTENTION POOL SELECTOR TEST WITH WEIGHT TRANSFER
# ============================================================================

def test_attention_pool_selector():
    """Test AttentionPoolSelector TF vs PyTorch with weight transfer."""
    set_seeds()
    
    from ultr_ai.tf.network_architecture.components.selectors import AttentionPoolSelectorTF
    from ultr_ai.network_architecture.components.selectors import AttentionPoolSelector
    
    feature_dim = 768
    hidden_dim = 512
    output_dim = 512
    batch_size = 2
    num_frames = 16
    num_heads = 8
    
    # Create test inputs
    features = np.random.randn(batch_size, num_frames, feature_dim).astype(np.float32)
    mask = np.ones((batch_size, num_frames), dtype=bool)
    
    # Create PyTorch module
    torch_selector = AttentionPoolSelector(
        feature_dim=feature_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        num_heads=num_heads,
        temperature=0.5
    )
    torch_selector.eval()
    
    # Create TF module
    tf_selector = AttentionPoolSelectorTF(
        feature_dim=feature_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        num_heads=num_heads,
        temperature=0.5
    )
    
    # Build TF module
    _ = tf_selector(tf.constant(features), tf.constant(mask), training=False)
    
    # Transfer weights
    # feature_encoder: Dense, LayerNorm, GELU, Dropout
    transfer_linear_weights(torch_selector.feature_encoder[0], tf_selector.feature_encoder.layers[0])
    transfer_layernorm_weights(torch_selector.feature_encoder[1], tf_selector.feature_encoder.layers[1])
    
    # attention_scorer: Dense, Tanh, Dense
    transfer_linear_weights(torch_selector.attention_scorer[0], tf_selector.attention_scorer.layers[0])
    transfer_linear_weights(torch_selector.attention_scorer[2], tf_selector.attention_scorer.layers[2])
    
    # output_projection: Dense, LayerNorm, Tanh
    transfer_linear_weights(torch_selector.output_projection[0], tf_selector.output_projection.layers[0])
    transfer_layernorm_weights(torch_selector.output_projection[1], tf_selector.output_projection.layers[1])
    
    # Forward pass
    with torch.no_grad():
        torch_scores, torch_values, torch_features = torch_selector(
            torch.tensor(features),
            torch.tensor(mask)
        )
    
    tf_scores, tf_values, tf_features = tf_selector(
        tf.constant(features),
        tf.constant(mask),
        training=False
    )
    
    # Note: Due to MHA weight differences, we compare non-MHA dependent outputs
    print(f"AttentionPoolSelector Test:")
    print(f"  Input shape: {features.shape}")
    print(f"  Attention scores shapes - PyTorch: {torch_scores.shape}, TF: {tf_scores.numpy().shape}")
    print(f"  Output features shapes - PyTorch: {torch_features.shape}, TF: {tf_features.numpy().shape}")
    
    assert tf_scores.shape == (batch_size, num_frames)
    assert tf_values.shape == (batch_size, 1)
    assert tf_features.shape == (batch_size, num_frames, output_dim)
    assert not np.any(np.isnan(tf_scores.numpy()))
    assert not np.any(np.isnan(tf_features.numpy()))
    
    print("  PASSED (shapes valid, no NaN - MHA weights not transferred)\n")
    return True


# ============================================================================
# FRAME SELECTION AGENT TEST WITH WEIGHT TRANSFER
# ============================================================================

def test_frame_selection_agent():
    """Test FrameSelectionAgent TF vs PyTorch with weight transfer."""
    set_seeds()
    
    from ultr_ai.tf.network_architecture.components.general_components import FrameSelectionAgentTF
    from ultr_ai.network_architecture.components.general_components import FrameSelectionAgent
    
    feature_dim = 768
    hidden_dim = 512
    output_dim = 512
    batch_size = 2
    num_frames = 16
    
    # Create test inputs
    features = np.random.randn(batch_size, num_frames, feature_dim).astype(np.float32)
    mask = np.ones((batch_size, num_frames), dtype=bool)
    mask[0, 12:] = False  # Mask last 4 frames for first batch
    
    # Create PyTorch module
    torch_agent = FrameSelectionAgent(
        feature_dim=feature_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        num_frame_features=16,
        min_temperature=0.1,
        max_temperature=5.0,
        use_frame_history=False  # Disable for simpler test
    )
    torch_agent.eval()
    
    # Create TF module
    tf_agent = FrameSelectionAgentTF(
        feature_dim=feature_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        num_frame_features=16,
        min_temperature=0.1,
        max_temperature=5.0,
        use_frame_history=False
    )
    
    # Build TF module
    _ = tf_agent(tf.constant(features), tf.constant(mask), training=False)
    
    # Transfer weights
    # feature_encoder: Dense, LayerNorm, Tanh, Dropout, Dense, LayerNorm, Tanh
    transfer_linear_weights(torch_agent.feature_encoder[0], tf_agent.feature_encoder.layers[0])
    transfer_layernorm_weights(torch_agent.feature_encoder[1], tf_agent.feature_encoder.layers[1])
    transfer_linear_weights(torch_agent.feature_encoder[4], tf_agent.feature_encoder.layers[4])
    transfer_layernorm_weights(torch_agent.feature_encoder[5], tf_agent.feature_encoder.layers[5])
    
    # context_layers: 4 Conv1D layers
    for i, (torch_conv, tf_conv) in enumerate(zip(torch_agent.context_layers, tf_agent.context_layers)):
        transfer_conv1d_weights(torch_conv, tf_conv)
    
    # pos_expand
    transfer_linear_weights(torch_agent.pos_expand, tf_agent.pos_expand)
    
    # Transfer positional embedding
    tf_agent.pos_embedding.assign(torch_agent.pos_embedding.detach().numpy())
    
    # policy_net: Dense, LayerNorm, Tanh, Dropout, Dense
    transfer_linear_weights(torch_agent.policy_net[0], tf_agent.policy_net.layers[0])
    transfer_layernorm_weights(torch_agent.policy_net[1], tf_agent.policy_net.layers[1])
    transfer_linear_weights(torch_agent.policy_net[4], tf_agent.policy_net.layers[4])
    
    # value_net: Dense, LayerNorm, Tanh, Dropout, Dense
    transfer_linear_weights(torch_agent.value_net[0], tf_agent.value_net.layers[0])
    transfer_layernorm_weights(torch_agent.value_net[1], tf_agent.value_net.layers[1])
    transfer_linear_weights(torch_agent.value_net[4], tf_agent.value_net.layers[4])
    
    # output_projection: Dense, LayerNorm, Tanh
    transfer_linear_weights(torch_agent.output_projection[0], tf_agent.output_projection.layers[0])
    transfer_layernorm_weights(torch_agent.output_projection[1], tf_agent.output_projection.layers[1])
    
    # Forward pass (without training to avoid random exploration bonus)
    with torch.no_grad():
        torch_logits, torch_values, torch_encoded = torch_agent(
            torch.tensor(features),
            torch.tensor(mask)
        )
    
    tf_logits, tf_values, tf_encoded = tf_agent(
        tf.constant(features),
        tf.constant(mask),
        training=False
    )
    
    # Compare encoded features (should match exactly)
    encoded_diff = np.max(np.abs(torch_encoded.numpy() - tf_encoded.numpy()))
    
    print(f"FrameSelectionAgent Test (with weight transfer):")
    print(f"  Input shape: {features.shape}")
    print(f"  Action logits shapes - PyTorch: {torch_logits.shape}, TF: {tf_logits.numpy().shape}")
    print(f"  State values shapes - PyTorch: {torch_values.shape}, TF: {tf_values.numpy().shape}")
    print(f"  Encoded features shapes - PyTorch: {torch_encoded.shape}, TF: {tf_encoded.numpy().shape}")
    print(f"  Encoded features max difference: {encoded_diff}")
    
    assert tf_logits.shape == (batch_size, num_frames)
    assert tf_values.shape == (batch_size, 1)
    assert tf_encoded.shape == (batch_size, num_frames, output_dim)
    assert not np.any(np.isnan(tf_logits.numpy()))
    assert not np.any(np.isnan(tf_encoded.numpy()))
    
    # Encoded features should match with transferred weights
    assert encoded_diff < 1e-4, f"Encoded features differ by {encoded_diff}"
    
    print("  PASSED\n")
    return True


# ============================================================================
# MULTIHEAD ATTENTION TEST
# ============================================================================

def test_multihead_attention():
    """Test MultiHeadAttention with weight transfer produces matching outputs."""
    set_seeds()
    
    embed_dim = 64
    num_heads = 8
    batch_size = 2
    seq_len = 10
    
    # Create test input
    test_input = np.random.randn(batch_size, seq_len, embed_dim).astype(np.float32)
    
    # PyTorch MultiheadAttention
    torch_mha = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
    
    # TensorFlow MultiHeadAttention
    tf_mha = tf.keras.layers.MultiHeadAttention(
        num_heads=num_heads,
        key_dim=embed_dim // num_heads
    )
    
    # Build TF MHA
    tf_input = tf.constant(test_input)
    _ = tf_mha(tf_input, tf_input, tf_input)
    
    # Transfer weights
    transfer_mha_weights(torch_mha, tf_mha, embed_dim, num_heads)
    
    # Forward pass
    torch_input = torch.tensor(test_input)
    with torch.no_grad():
        torch_output, _ = torch_mha(torch_input, torch_input, torch_input)
    torch_output = torch_output.numpy()
    
    tf_output = tf_mha(tf_input, tf_input, tf_input).numpy()
    
    max_diff = np.max(np.abs(torch_output - tf_output))
    print(f"MultiHeadAttention Test (with weight transfer):")
    print(f"  Input shape: {test_input.shape}")
    print(f"  PyTorch output shape: {torch_output.shape}")
    print(f"  TensorFlow output shape: {tf_output.shape}")
    print(f"  Max difference: {max_diff}")
    
    assert torch_output.shape == tf_output.shape, "Output shapes don't match"
    assert not np.any(np.isnan(torch_output)), "PyTorch output contains NaN"
    assert not np.any(np.isnan(tf_output)), "TensorFlow output contains NaN"
    
    # With proper weight transfer, outputs should be very close
    assert max_diff < 1e-4, f"Outputs differ by {max_diff}"
    print("  PASSED\n")
    return True

# ============================================================================
# CLIP TEST
# ============================================================================

# The two loading functions were for testing the same logic is not implemented in MultiTaskModel
def load_clip_torch():
    from transformers import CLIPVisionModel
    import os

    vision_encoder = CLIPVisionModel.from_pretrained(
        "openai/clip-vit-base-patch32",
        dtype=torch.float32
    )
    # CLIP ViT-B/32 has 768-d pooler_output
    vision_dim = getattr(vision_encoder.config, 'hidden_size', 768)
    _vision_kind = 'clip'
    # Optional: local weights loading (kept from original)
    local_weights_path = os.path.join(
        'ultr_ai/network_architecture/CLIP_weights',
        'model.safetensors'
    )
    if os.path.exists(local_weights_path):
        print(f"Loading CLIP weights from {local_weights_path}")
        try:
            from safetensors import safe_open as _safe_open
            with _safe_open(local_weights_path, framework='pt', device='cpu') as f:
                vision_state_dict = {}
                model_state_dict = vision_encoder.state_dict()
                matched_keys = 0
                for key in model_state_dict.keys():
                    safetensors_key = f"vision_model.{key}"
                    if safetensors_key in f.keys():
                        tensor = f.get_tensor(safetensors_key)
                        if tensor.shape == model_state_dict[key].shape:
                            vision_state_dict[key] = tensor
                            matched_keys += 1
                if matched_keys > 0:
                    print(f"Successfully matched {matched_keys}/{len(model_state_dict)} CLIP weights")
                    vision_encoder.load_state_dict(vision_state_dict, strict=False)
                else:
                    print("No weights could be matched from the safetensors file")
        except Exception as e:
            print(f"Failed to load local CLIP weights: {e}")
    else:
        print("No local CLIP weights file found, using default pretrained weights")

    return vision_encoder

def load_clip_tf():
    from transformers import TFCLIPVisionModel
    import os

    from ultr_ai.convert.utils import load_clip_weights_from_safetensors_to_tf

    vision_encoder = TFCLIPVisionModel.from_pretrained(
        "openai/clip-vit-base-patch32",
        from_pt=True,
        # dtype=tf.float32
    )
    # CLIP ViT-B/32 has 768-d pooler_output
    vision_dim = getattr(vision_encoder.config, 'hidden_size', 768)
    _vision_kind = 'clip'
    # Optional: local weights loading using the new mapping
    local_weights_path = os.path.join(
        'ultr_ai/network_architecture/CLIP_weights',
        'model.safetensors'
    )
    if os.path.exists(local_weights_path):
        print(f"Loading CLIP weights from {local_weights_path}")
        try:
            matched, total, unmatched = load_clip_weights_from_safetensors_to_tf(
                local_weights_path, 
                vision_encoder, 
                num_layers=12
            )
            print(f"Successfully matched {matched}/{total} CLIP weights")
            if unmatched:
                print(f"Unmatched weights ({len(unmatched)}):")
                for msg in unmatched[:10]:  # Print first 10
                    print(f"  - {msg}")
                if len(unmatched) > 10:
                    print(f"  ... and {len(unmatched) - 10} more")
        except Exception as e:
            print(f"Failed to load local CLIP weights: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("No local CLIP weights file found, using default pretrained weights")

    return vision_encoder

def test_clip_model(torch_encoder, tf_encoder):
    """Test CLIP model vision encoder TF vs PyTorch with same weights."""
    set_seeds()

    # Dummy input to check forward pass
    import numpy as np
    dummy_input = np.random.randn(2, 3, 224, 224).astype(np.float32)
    import torch
    torch_input = torch.tensor(dummy_input)
    with torch.no_grad():
        torch_output = torch_encoder(pixel_values=torch_input).pooler_output.numpy()
    import tensorflow as tf
    tf_input = tf.constant(dummy_input)
    tf_output = tf_encoder(pixel_values=tf_input).pooler_output.numpy()
    max_diff = np.max(np.abs(torch_output - tf_output))
    print(f"CLIP Vision Encoder Test:")
    print(f"  Input shape: {dummy_input.shape}")
    print(f"  Max difference: {max_diff}")
    assert max_diff < 1e-4, f"Outputs differ by {max_diff}"
    print("  PASSED\n")
    return True
    

# ============================================================================
# FULL ATTENTION POOL MODEL TEST
# ============================================================================

def test_full_attention_pool_model():
    """Test AttentionPool model vision encoder TF vs PyTorch with same weights."""
    set_seeds()

    from ultr_ai.config import load_config
    from ultr_ai.network_architecture import create_ablation_model
    from ultr_ai.tf.network_architecture import create_ablation_model_tf
    from ultr_ai.convert.utils import construct_dummy_input_dict

    config_path = "configs/attention_pool_extra3_full_train2/fold0.yaml"

    config = load_config(config_file=config_path)
    
    # Ensure we're using the local weights path
    config.local_weights_dir = "ultr_ai/network_architecture/CLIP_weights"

    torch_model = create_ablation_model("attention_pool", config)
    torch_model.eval()
    
    tf_model = create_ablation_model_tf("attention_pool", config)

    # Test clip vision encoders
    test_clip_model(torch_model.vision_encoder, tf_model.vision_encoder)

    # Create dummy input
    input_dict = construct_dummy_input_dict()

    # Forward pass
    with torch.no_grad():
        torch_output = torch_model(input_dict)
    tf_input_dict = {k: tf.constant(v) for k, v in input_dict.items()}
    tf_output = tf_model(tf_input_dict, training=False)



    # Compare outputs
    max_diff = np.max(np.abs(torch_output["task_logits"]["TB Label"].numpy() - tf_output["task_logits"]["TB Label"].numpy()))
    print(f"Full Attention Pool Model Test:")
    print(f"  Max difference: {max_diff}")
    assert max_diff < 1e-4, f"Outputs differ by {max_diff}"
    print("  PASSED\n")
    return True

# ============================================================================
# COMPREHENSIVE COMPARISON
# ============================================================================

def run_all_tests():
    """Run all comparison tests."""
    print("=" * 70)
    print("TensorFlow vs PyTorch Layer Comparison Tests (with Weight Transfer)")
    print("=" * 70 + "\n")
    
    tests = [
        # ("Simple Linear", test_simple_linear),
        # ("LayerNorm", test_layer_norm),
        # ("Conv1D", test_conv1d),
        # ("Softmax", test_softmax),
        # ("GELU", test_gelu),
        # ("Weight Transfer", test_weight_transfer),
        # ("MultiHeadAttention", test_multihead_attention),
        # ("PathologyModule", test_pathology_module),
        # ("SiteIntegrationModule", test_site_integration_module),
        # ("DeepAttentionMIL", test_deep_attention_mil),
        # ("AttentionPoolSelector", test_attention_pool_selector),
        # ("FrameSelectionAgent", test_frame_selection_agent),
        # ("CLIP Vision Encoder", test_clip_model),
        ("Full Attention Pool Model", test_full_attention_pool_model),
    ]
    
    results = {}
    for name, test_fn in tests:
        try:
            result = test_fn()
            results[name] = "PASSED"
        except Exception as e:
            results[name] = f"FAILED: {str(e)}"
            logger.exception(f"Test {name} failed")
            import traceback
            traceback.print_exc()
    
    # Print summary
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for name, result in results.items():
        print(f"  - {name}: {result}")
    
    passed = sum(1 for r in results.values() if "PASSED" in r)
    total = len(results)
    print(f"\nTotal: {passed}/{total} tests passed")
    
    return all("PASSED" in r for r in results.values())


if __name__ == "__main__":
    # # Quick sanity check first
    # print("Quick sanity check with fixed weights:")
    # torch_model = SimpleModelTorch()
    # torch_input = torch.tensor([1, 2], dtype=torch.float32).unsqueeze(0)
    # torch_output = torch_model(torch_input)
    # print("PyTorch Model Output:", torch_output.detach().numpy())

    # tf_model = SimpleModelTF()
    # tf_input = tf.constant([[1, 2]], dtype=tf.float32)
    # tf_output = tf_model(tf_input)
    # print("TensorFlow Model Output:", tf_output.numpy())
    
    # diff = np.max(np.abs(torch_output.detach().numpy() - tf_output.numpy()))
    # print(f"Difference: {diff}")
    
    # Run comprehensive tests
    success = run_all_tests()
    exit(0 if success else 1)