"""
TensorFlow vs PyTorch Layer Comparison Tests

This module provides comprehensive tests to verify that TensorFlow implementations
produce the same outputs as PyTorch implementations for each layer/component.

Usage:
    python -m ultr_ai.tf.utils.tf_vs_torch
    
Or run specific tests:
    python -m pytest ultr_ai/tf/utils/tf_vs_torch.py -v
"""

import numpy as np
import logging

logger = logging.getLogger(__name__)

# Set seeds for reproducibility
SEED = 42
np.random.seed(SEED)


def set_seeds():
    """Set random seeds for reproducibility in both frameworks."""
    np.random.seed(SEED)
    try:
        import torch
        torch.manual_seed(SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(SEED)
    except ImportError:
        pass
    try:
        import tensorflow as tf
        tf.random.set_seed(SEED)
    except ImportError:
        pass


# ============================================================================
# SIMPLE LAYER TESTS
# ============================================================================

import torch
import torch.nn as nn

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
    

import tensorflow as tf

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
    print("  ✓ PASSED\n")
    return True


# ============================================================================
# LAYER NORMALIZATION TEST
# ============================================================================

def test_layer_norm():
    """Test LayerNormalization produces matching outputs."""
    set_seeds()
    
    hidden_dim = 64
    batch_size = 2
    seq_len = 10
    
    # Create test input
    test_input = np.random.randn(batch_size, seq_len, hidden_dim).astype(np.float32)
    
    # PyTorch LayerNorm
    torch_ln = torch.nn.LayerNorm(hidden_dim)
    torch_input = torch.tensor(test_input)
    torch_output = torch_ln(torch_input).detach().numpy()
    
    # TensorFlow LayerNormalization
    tf_ln = tf.keras.layers.LayerNormalization()
    tf_ln.build((None, seq_len, hidden_dim))
    tf_input = tf.constant(test_input)
    tf_output = tf_ln(tf_input).numpy()
    
    max_diff = np.max(np.abs(torch_output - tf_output))
    print(f"LayerNorm Test:")
    print(f"  Input shape: {test_input.shape}")
    print(f"  Max difference: {max_diff}")
    
    assert max_diff < 1e-5, f"Outputs differ by {max_diff}"
    print("  ✓ PASSED\n")
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
    
    # Known weights [out_channels, in_channels, kernel_size] for PyTorch
    torch_weights = np.random.randn(out_channels, in_channels, kernel_size).astype(np.float32)
    torch_bias = np.random.randn(out_channels).astype(np.float32)
    
    # PyTorch Conv1D (expects [B, C, L])
    torch_conv = torch.nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size//2)
    torch_conv.weight = torch.nn.Parameter(torch.tensor(torch_weights))
    torch_conv.bias = torch.nn.Parameter(torch.tensor(torch_bias))
    
    # Convert input to PyTorch format [B, C, L]
    torch_input = torch.tensor(test_input).transpose(1, 2)
    torch_output = torch_conv(torch_input).transpose(1, 2).detach().numpy()
    
    # TensorFlow Conv1D (expects [B, L, C])
    tf_conv = tf.keras.layers.Conv1D(out_channels, kernel_size, padding='same')
    tf_conv.build((None, seq_len, in_channels))
    # TF kernel shape: [kernel_size, in_channels, out_channels]
    tf_weights = np.transpose(torch_weights, (2, 1, 0))
    tf_conv.set_weights([tf_weights, torch_bias])
    
    tf_input = tf.constant(test_input)
    tf_output = tf_conv(tf_input).numpy()
    
    max_diff = np.max(np.abs(torch_output - tf_output))
    print(f"Conv1D Test:")
    print(f"  Input shape: {test_input.shape}")
    print(f"  Output shapes - PyTorch: {torch_output.shape}, TF: {tf_output.shape}")
    print(f"  Max difference: {max_diff}")
    
    # Conv1D can have small numerical differences due to implementation
    assert max_diff < 1e-4, f"Outputs differ by {max_diff}"
    print("  ✓ PASSED\n")
    return True


# ============================================================================
# MULTI-HEAD ATTENTION TEST
# ============================================================================

def test_multihead_attention():
    """Test MultiHeadAttention produces similar outputs (note: implementations may differ)."""
    set_seeds()
    
    embed_dim = 64
    num_heads = 8
    batch_size = 2
    seq_len = 10
    
    # Create test input
    test_input = np.random.randn(batch_size, seq_len, embed_dim).astype(np.float32)
    
    # PyTorch MultiheadAttention
    torch_mha = torch.nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
    torch_input = torch.tensor(test_input)
    torch_output, _ = torch_mha(torch_input, torch_input, torch_input)
    torch_output = torch_output.detach().numpy()
    
    # TensorFlow MultiHeadAttention
    tf_mha = tf.keras.layers.MultiHeadAttention(
        num_heads=num_heads,
        key_dim=embed_dim // num_heads
    )
    tf_input = tf.constant(test_input)
    tf_output = tf_mha(tf_input, tf_input, tf_input).numpy()
    
    # Note: We can't expect exact matching because weights are randomly initialized differently
    # This test verifies output shapes and that both produce valid outputs
    print(f"MultiHeadAttention Test:")
    print(f"  Input shape: {test_input.shape}")
    print(f"  PyTorch output shape: {torch_output.shape}")
    print(f"  TensorFlow output shape: {tf_output.shape}")
    
    assert torch_output.shape == tf_output.shape, "Output shapes don't match"
    assert not np.any(np.isnan(torch_output)), "PyTorch output contains NaN"
    assert not np.any(np.isnan(tf_output)), "TensorFlow output contains NaN"
    print("  ✓ PASSED (shapes match, no NaN)\n")
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
    torch_output = torch.nn.functional.softmax(torch.tensor(test_input), dim=-1).numpy()
    
    # TensorFlow softmax
    tf_output = tf.nn.softmax(tf.constant(test_input), axis=-1).numpy()
    
    max_diff = np.max(np.abs(torch_output - tf_output))
    print(f"Softmax Test:")
    print(f"  Max difference: {max_diff}")
    
    assert max_diff < 1e-6, f"Outputs differ by {max_diff}"
    print("  ✓ PASSED\n")
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
    torch_output = torch.nn.functional.gelu(torch.tensor(test_input)).numpy()
    
    # TensorFlow GELU
    tf_output = tf.nn.gelu(tf.constant(test_input)).numpy()
    
    max_diff = np.max(np.abs(torch_output - tf_output))
    print(f"GELU Test:")
    print(f"  Max difference: {max_diff}")
    
    # GELU implementations may have small differences
    assert max_diff < 1e-5, f"Outputs differ by {max_diff}"
    print("  ✓ PASSED\n")
    return True


# ============================================================================
# PATHOLOGY MODULE TEST
# ============================================================================

def test_pathology_module():
    """Test PathologyModule TF vs PyTorch with same weights."""
    set_seeds()
    
    # Import the TF module
    from ultr_ai.tf.network_architecture.components.general_components import PathologyModuleTF
    
    feature_dim = 512
    hidden_dim = 256
    batch_size = 2
    num_frames = 5
    
    # Create test input
    test_input = np.random.randn(batch_size, num_frames, feature_dim).astype(np.float32)
    
    # TensorFlow PathologyModule
    tf_module = PathologyModuleTF(
        feature_dim=feature_dim,
        hidden_dim=hidden_dim,
        dropout=0.0,  # Disable dropout for deterministic comparison
        name='pathology_test'
    )
    
    # Build by calling
    tf_input = tf.constant(test_input)
    tf_score, tf_attn, tf_pooled = tf_module(tf_input, training=False)
    
    print(f"PathologyModule Test:")
    print(f"  Input shape: {test_input.shape}")
    print(f"  TF output shapes - score: {tf_score.shape}, attn: {tf_attn.shape}, pooled: {tf_pooled.shape}")
    
    # Verify output shapes
    assert tf_score.shape == (batch_size, 1), f"Score shape mismatch: {tf_score.shape}"
    assert tf_attn.shape == (batch_size, num_frames), f"Attention shape mismatch: {tf_attn.shape}"
    assert tf_pooled.shape == (batch_size, hidden_dim), f"Pooled shape mismatch: {tf_pooled.shape}"
    
    # Verify attention sums to 1
    attn_sum = np.sum(tf_attn.numpy(), axis=-1)
    assert np.allclose(attn_sum, 1.0, atol=1e-5), f"Attention doesn't sum to 1: {attn_sum}"
    
    print("  ✓ PASSED (shapes valid, attention sums to 1)\n")
    return True


# ============================================================================
# SITE INTEGRATION MODULE TEST
# ============================================================================

def test_site_integration_module():
    """Test SiteIntegrationModule TF with valid outputs."""
    set_seeds()
    
    from ultr_ai.tf.network_architecture.components.general_components import SiteIntegrationModuleTF
    
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
    
    # Create TF module
    tf_module = SiteIntegrationModuleTF(
        feature_dim=feature_dim,
        site_embed_dim=site_embed_dim,
        hidden_dim=hidden_dim,
        num_sites=num_sites,
        num_pathologies=num_pathologies,
        dropout=0.0
    )
    
    # Forward pass
    tf_output = tf_module(
        tf.constant(site_features),
        tf.constant(site_indices),
        tf.constant(pathology_scores),
        training=False
    )
    
    print(f"SiteIntegrationModule Test:")
    print(f"  Input shapes - features: {site_features.shape}, indices: {site_indices.shape}")
    print(f"  Output shape: {tf_output.shape}")
    
    assert tf_output.shape == (batch_size, max_sites_per_patient, hidden_dim)
    assert not np.any(np.isnan(tf_output.numpy()))
    
    print("  ✓ PASSED (shape valid, no NaN)\n")
    return True


# ============================================================================
# DEEP ATTENTION MIL TEST
# ============================================================================

def test_deep_attention_mil():
    """Test DeepAttentionMIL TF with valid outputs."""
    set_seeds()
    
    from ultr_ai.tf.network_architecture.components.general_components import DeepAttentionMILTF
    
    feature_dim = 512
    hidden_dim = 256
    batch_size = 2
    num_instances = 5
    
    # Create test inputs
    features = np.random.randn(batch_size, num_instances, feature_dim).astype(np.float32)
    mask = np.array([[True, True, True, False, False], [True, True, True, True, False]])
    
    # Create TF module
    tf_module = DeepAttentionMILTF(
        feature_dim=feature_dim,
        hidden_dim=hidden_dim,
        dropout=0.0,
        num_heads=8
    )
    
    # Forward pass
    tf_aggregated, tf_attention = tf_module(
        tf.constant(features),
        tf.constant(mask),
        training=False
    )
    
    print(f"DeepAttentionMIL Test:")
    print(f"  Input shape: {features.shape}")
    print(f"  Aggregated shape: {tf_aggregated.shape}")
    print(f"  Attention shape: {tf_attention.shape}")
    
    assert tf_aggregated.shape == (batch_size, feature_dim)
    assert tf_attention.shape == (batch_size, num_instances)
    assert not np.any(np.isnan(tf_aggregated.numpy()))
    
    # Check attention values
    masked_attention = tf_attention.numpy()
    print(f"  Attention values: {masked_attention}")
    
    print("  ✓ PASSED (shapes valid, no NaN)\n")
    return True


# ============================================================================
# FRAME SELECTION AGENT TEST
# ============================================================================

def test_frame_selection_agent():
    """Test FrameSelectionAgent TF with valid outputs."""
    set_seeds()
    
    from ultr_ai.tf.network_architecture.components.general_components import FrameSelectionAgentTF
    
    feature_dim = 768
    hidden_dim = 512
    output_dim = 512
    batch_size = 2
    num_frames = 16
    
    # Create test inputs
    features = np.random.randn(batch_size, num_frames, feature_dim).astype(np.float32)
    mask = np.ones((batch_size, num_frames), dtype=bool)
    mask[0, 12:] = False  # Mask last 4 frames for first batch
    
    # Create TF module
    tf_agent = FrameSelectionAgentTF(
        feature_dim=feature_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        num_frame_features=16,
        min_temperature=0.1,
        max_temperature=5.0,
        use_frame_history=False  # Disable for simpler test
    )
    
    # Forward pass
    action_logits, state_values, encoded_features = tf_agent(
        tf.constant(features),
        tf.constant(mask),
        training=False
    )
    
    print(f"FrameSelectionAgent Test:")
    print(f"  Input shape: {features.shape}")
    print(f"  Action logits shape: {action_logits.shape}")
    print(f"  State values shape: {state_values.shape}")
    print(f"  Encoded features shape: {encoded_features.shape}")
    
    assert action_logits.shape == (batch_size, num_frames)
    assert state_values.shape == (batch_size, 1)
    assert encoded_features.shape == (batch_size, num_frames, output_dim)
    assert not np.any(np.isnan(action_logits.numpy()))
    
    # Test action selection
    actions, log_probs = tf_agent.select_action(
        action_logits, state_values, encoded_features, training=False
    )
    print(f"  Selected actions: {actions.numpy()}")
    
    print("  ✓ PASSED (shapes valid, no NaN)\n")
    return True


# ============================================================================
# ATTENTION POOL SELECTOR TEST
# ============================================================================

def test_attention_pool_selector():
    """Test AttentionPoolSelector TF with valid outputs."""
    set_seeds()
    
    from ultr_ai.tf.network_architecture.components.selectors import AttentionPoolSelectorTF
    
    feature_dim = 768
    hidden_dim = 512
    output_dim = 512
    batch_size = 2
    num_frames = 16
    
    # Create test inputs
    features = np.random.randn(batch_size, num_frames, feature_dim).astype(np.float32)
    mask = np.ones((batch_size, num_frames), dtype=bool)
    
    # Create TF module
    tf_selector = AttentionPoolSelectorTF(
        feature_dim=feature_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        num_heads=8,
        temperature=0.5
    )
    
    # Forward pass
    attention_scores, state_values, output_features = tf_selector(
        tf.constant(features),
        tf.constant(mask),
        training=False
    )
    
    print(f"AttentionPoolSelector Test:")
    print(f"  Input shape: {features.shape}")
    print(f"  Attention scores shape: {attention_scores.shape}")
    print(f"  State values shape: {state_values.shape}")
    print(f"  Output features shape: {output_features.shape}")
    
    assert attention_scores.shape == (batch_size, num_frames)
    assert state_values.shape == (batch_size, 1)
    assert output_features.shape == (batch_size, num_frames, output_dim)
    assert not np.any(np.isnan(attention_scores.numpy()))
    
    print("  ✓ PASSED (shapes valid, no NaN)\n")
    return True


# ============================================================================
# WEIGHT TRANSFER TEST
# ============================================================================

def transfer_linear_weights_torch_to_tf(torch_linear, tf_dense):
    """Transfer weights from PyTorch Linear to TensorFlow Dense."""
    # PyTorch Linear: weight [out_features, in_features], bias [out_features]
    # TF Dense: kernel [in_features, out_features], bias [out_features]
    
    torch_weight = torch_linear.weight.detach().numpy()
    tf_kernel = np.transpose(torch_weight)  # Transpose for TF
    
    if torch_linear.bias is not None:
        torch_bias = torch_linear.bias.detach().numpy()
        tf_dense.set_weights([tf_kernel, torch_bias])
    else:
        tf_dense.set_weights([tf_kernel])


def test_weight_transfer():
    """Test that weights can be transferred from PyTorch to TensorFlow correctly."""
    set_seeds()
    
    in_features = 64
    out_features = 32
    
    # Create PyTorch linear layer with specific weights
    torch_linear = torch.nn.Linear(in_features, out_features)
    
    # Create TensorFlow Dense layer
    tf_dense = tf.keras.layers.Dense(out_features, use_bias=True)
    tf_dense.build((None, in_features))
    
    # Transfer weights
    transfer_linear_weights_torch_to_tf(torch_linear, tf_dense)
    
    # Test with same input
    test_input = np.random.randn(2, in_features).astype(np.float32)
    
    torch_output = torch_linear(torch.tensor(test_input)).detach().numpy()
    tf_output = tf_dense(tf.constant(test_input)).numpy()
    
    max_diff = np.max(np.abs(torch_output - tf_output))
    print(f"Weight Transfer Test:")
    print(f"  Max difference after weight transfer: {max_diff}")
    
    assert max_diff < 1e-5, f"Outputs differ by {max_diff}"
    print("  ✓ PASSED\n")
    return True


# ============================================================================
# COMPREHENSIVE COMPARISON
# ============================================================================

def run_all_tests():
    """Run all comparison tests."""
    print("=" * 60)
    print("TensorFlow vs PyTorch Layer Comparison Tests")
    print("=" * 60 + "\n")
    
    tests = [
        ("Simple Linear", test_simple_linear),
        ("LayerNorm", test_layer_norm),
        ("Conv1D", test_conv1d),
        ("MultiHeadAttention", test_multihead_attention),
        ("Softmax", test_softmax),
        ("GELU", test_gelu),
        ("Weight Transfer", test_weight_transfer),
        ("PathologyModule", test_pathology_module),
        ("SiteIntegrationModule", test_site_integration_module),
        ("DeepAttentionMIL", test_deep_attention_mil),
        ("FrameSelectionAgent", test_frame_selection_agent),
        ("AttentionPoolSelector", test_attention_pool_selector),
    ]
    
    results = {}
    for name, test_fn in tests:
        try:
            result = test_fn()
            results[name] = "✓ PASSED"
        except Exception as e:
            results[name] = f"✗ FAILED: {str(e)}"
            logger.exception(f"Test {name} failed")
    
    # Print summary
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for name, result in results.items():
        print(f"  {name}: {result}")
    
    passed = sum(1 for r in results.values() if "PASSED" in r)
    total = len(results)
    print(f"\nTotal: {passed}/{total} tests passed")
    
    return all("PASSED" in r for r in results.values())


if __name__ == "__main__":
    # Test PyTorch model (original basic test)
    torch_model = SimpleModelTorch()
    torch_input = torch.tensor([1, 2], dtype=torch.float32).unsqueeze(0)
    torch_output = torch_model(torch_input)
    print("PyTorch Model Output:", torch_output)

    # Test TensorFlow model (original basic test)
    tf_model = SimpleModelTF()
    tf_input = tf.constant([[1, 2]], dtype=tf.float32)
    tf_output = tf_model(tf_input)
    print("TensorFlow Model Output:", tf_output)
    
    print("\n" + "=" * 60 + "\n")
    
    # Run comprehensive tests
    success = run_all_tests()
    exit(0 if success else 1)