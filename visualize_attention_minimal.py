"""
Simple Script to Visualize Top-3 Frames Selected by Attention
==============================================================

Shows the 3 ultrasound frames with HIGHEST attention scores.
"""

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
import argparse
from pathlib import Path

# Add paths
sys.path.insert(0, ".")

from config import load_config
from NetworkArchitecture.ablation_models import create_ablation_model


def load_and_preprocess_video(video_path, target_size=(224, 224), max_frames=None):
    """Load video and preprocess for model input."""
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Resize
        frame = cv2.resize(frame, target_size)
        # To CHW format
        frame = np.transpose(frame, (2, 0, 1))
        frames.append(frame)
        
        if max_frames and len(frames) >= max_frames:
            break
    
    cap.release()
    
    if len(frames) == 0:
        raise ValueError(f"No frames loaded from {video_path}")
    
    frames = np.stack(frames, axis=0)  # [T, C, H, W]
    return frames


def get_top_k_frames(model, frames, device, k=3):
    """Get the k frames with HIGHEST attention scores (matching model's process_site logic)."""
    # Normalize and convert to tensor
    video = torch.from_numpy(frames).float() / 255.0
    video = video.unsqueeze(0).to(device)  # [1, T, C, H, W]
    
    # All frames are valid
    mask = torch.ones(1, video.shape[1], dtype=torch.bool, device=device)
    
    with torch.no_grad():
        # Extract features using CLIP
        features = model._extract_vision_features(video)  # [1, T, D]
        
        # Get attention logits from attention pooling selector
        attention_logits, _, _ = model.frame_selector(features, mask)
        
        # EXACTLY match what process_site does (line 1186 in CLIP_DRL_Aug11.py):
        # Get top 3 indices directly from logits (NO softmax)
        k = min(k, attention_logits.shape[1])
        scores, indices = torch.topk(attention_logits[0], k=k)
        
        # Also return all scores for statistics
        all_scores = attention_logits[0].cpu().numpy()
        
    return indices.cpu().numpy(), scores.cpu().numpy(), all_scores


def visualize_top_3_frames(frames, indices, scores, save_path):
    """Show just the 3 frames with highest attention."""
    
    # Create output directory if needed
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for i, (idx, score) in enumerate(zip(indices, scores)):
        ax = axes[i]
        
        # Get frame and convert to displayable format [C, H, W] -> [H, W, C]
        frame = frames[idx]
        img = np.transpose(frame, (1, 2, 0))
        
        # Normalize to [0, 1] if needed
        if img.max() > 1:
            img = img / 255.0
        
        ax.imshow(img)
        ax.set_title(f'Rank #{i+1}\nFrame {idx} | Score: {score:.4f}', 
                    fontsize=14, fontweight='bold')
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {save_path}")
    
    return fig


def main():
    parser = argparse.ArgumentParser(description='Visualize top-3 attention frames')
    parser.add_argument('--video', type=str, 
                       default='/capstor/scratch/cscs/mbarbiere/ultr-ai/LusBeninVideos/25-731_QPIG_15_1.mp4',
                       help='Path to video file (default: first video found)')
    parser.add_argument('--checkpoint', type=str,
                       default='/capstor/store/cscs/swissai/a127/ultr-ai/ablation_results/attention_pool_extra3/fold0/checkpoint_best_metric_0.9123.pth')
    parser.add_argument('--output', type=str, default='visualization_outputs/top3_attention_frames.png',
                       help='Output filename')
    parser.add_argument('--device', type=str, default='cuda')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.video):
        print(f"❌ Video not found: {args.video}")
        return
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    print("="*70)
    print(f"Video: {Path(args.video).name}")
    
    # Load video
    print("Loading video...")
    frames = load_and_preprocess_video(args.video)
    print(f"  {len(frames)} frames loaded")
    
    # Load model
    print("Loading model...")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    
    from types import SimpleNamespace
    config = SimpleNamespace(**checkpoint['config'])
    model_type = getattr(config, 'model_type', 'attention_pool')
    model = create_ablation_model(model_type, config).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print("  Model loaded")
    
    # Get top-3 frames with HIGHEST attention
    print("Computing attention scores...")
    top_indices, top_scores, all_scores = get_top_k_frames(model, frames, device, k=3)
    
    print(f"\nAttention score statistics:")
    print(f"  Total frames: {len(all_scores)}")
    print(f"  Min score: {all_scores.min():.4f}")
    print(f"  Max score: {all_scores.max():.4f}")
    print(f"  Mean score: {all_scores.mean():.4f}")
    print(f"  Std score: {all_scores.std():.4f}")
    
    print(f"\nTop-3 frames with HIGHEST attention:")
    for i, (idx, score) in enumerate(zip(top_indices, top_scores)):
        print(f"  #{i+1}: Frame {idx:3d} -> Score {score:.4f}")
    
    # Visualize
    print(f"\nCreating visualization...")
    visualize_top_3_frames(frames, top_indices, top_scores, args.output)
    
    print("="*70)
    print("✅ DONE!")
    

if __name__ == '__main__':
    main()
