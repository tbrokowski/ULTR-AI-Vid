"""
Simple Script to Visualize Top-3 Frames Selected by Attention
==============================================================

Shows the 3 ultrasound frames with HIGHEST attention scores.

NOTE: The model uses Gumbel-Softmax during TRAINING for differentiable frame selection,
but during INFERENCE (eval mode) it uses deterministic torch.topk() for hard selection.
This script uses eval mode, so it visualizes the actual frames selected by topk.

BATCH MODE: Use --batch-mode to efficiently process one video across multiple folds,
loading the video only once instead of multiple times.
"""

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
import argparse
import csv
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
    """
    Get the k frames with HIGHEST attention scores.
    
    This matches the model's inference behavior:
    - During TRAINING: model uses Gumbel-Softmax (soft, differentiable selection)
    - During INFERENCE: model uses torch.topk (hard, deterministic selection)
    
    Since we call model.eval(), we get the deterministic topk behavior.
    """
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
        
        # EXACTLY match what process_site does during INFERENCE (line 1200 in CLIP_DRL_Aug11.py):
        # Use torch.topk for hard selection (this is what model does in eval mode)
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


def compute_average_metrics(csv_path):
    """
    Compute average metrics across all videos in the CSV.
    
    Process:
    1. For each video, average across folds to get per-video statistics
    2. Then compute inter-video statistics (mean, std, min, max across videos)
    
    Returns:
        dict: Contains inter-video statistics and per-video data
    """
    if not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0:
        return None
    
    # Read all data and group by video
    from collections import defaultdict
    video_fold_data = defaultdict(list)
    
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            video_name = row['video_name']
            fold_stats = {
                'mean': float(row['mean_attention']),
                'std': float(row['std_attention']),
                'min': float(row['min_attention']),
                'max': float(row['max_attention'])
            }
            video_fold_data[video_name].append(fold_stats)
    
    if not video_fold_data:
        return None
    
    # Step 1: Average across folds for each video
    intra_video_stats = {}
    for video, fold_stats in video_fold_data.items():
        # Average across folds for this video
        video_stats = {k: np.mean([fs[k] for fs in fold_stats]) for k in fold_stats[0]}
        intra_video_stats[video] = video_stats
    
    # Step 2: Compute inter-video statistics (with min and max)
    inter_video_stats = {
        metric: {
            "mean": np.mean([v[metric] for v in intra_video_stats.values()]),
            "std": np.std([v[metric] for v in intra_video_stats.values()]),
            "min": np.min([v[metric] for v in intra_video_stats.values()]),
            "max": np.max([v[metric] for v in intra_video_stats.values()])
        }
        for metric in ["mean", "min", "max", "std"]
    }
    
    return {
        'num_videos': len(intra_video_stats),
        'inter_video_stats': inter_video_stats,
        'intra_video_stats': intra_video_stats  # Include per-video data
    }


def save_summary_metrics(csv_path, summary_path):
    """
    Compute and save summary metrics to a separate file.
    Combines inter-video statistics and per-video metrics in one CSV.
    
    Args:
        csv_path: Path to the detailed CSV file
        summary_path: Path to save the summary metrics
    """
    metrics = compute_average_metrics(csv_path)
    if not metrics:
        print("⚠️  No data available for summary metrics")
        return
    
    # Create output directory if needed
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    
    with open(summary_path, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Section 1: Inter-video statistics
        writer.writerow(['INTER-VIDEO STATISTICS (across all videos)'])
        writer.writerow(['metric', 'mean', 'std', 'min', 'max'])
        
        stats = metrics['inter_video_stats']
        for metric in ['mean', 'std', 'min', 'max']:
            writer.writerow([
                f'intra_video_{metric}',
                f"{stats[metric]['mean']:.6f}",
                f"{stats[metric]['std']:.6f}",
                f"{stats[metric]['min']:.6f}",
                f"{stats[metric]['max']:.6f}"
            ])
        
        # Blank line separator
        writer.writerow([])
        
        # Section 2: Per-video statistics (averaged across folds)
        writer.writerow(['PER-VIDEO STATISTICS (averaged across folds)'])
        writer.writerow(['video_name', 'mean', 'std', 'min', 'max'])
        
        for video, video_stats in sorted(metrics['intra_video_stats'].items()):
            writer.writerow([
                video,
                f"{video_stats['mean']:.6f}",
                f"{video_stats['std']:.6f}",
                f"{video_stats['min']:.6f}",
                f"{video_stats['max']:.6f}"
            ])
    
    print(f"✅ Summary metrics saved: {summary_path}")


def save_per_video_metrics(csv_path, per_video_path):
    """
    Save per-video metrics (averaged across folds) to a separate CSV.
    
    Args:
        csv_path: Path to the detailed CSV file
        per_video_path: Path to save the per-video metrics
    """
    if not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0:
        return
    
    # Read all data and group by video
    from collections import defaultdict
    video_fold_data = defaultdict(list)
    
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            video_name = row['video_name']
            fold_stats = {
                'mean': float(row['mean_attention']),
                'std': float(row['std_attention']),
                'min': float(row['min_attention']),
                'max': float(row['max_attention'])
            }
            video_fold_data[video_name].append(fold_stats)
    
    if not video_fold_data:
        return
    
    # Average across folds for each video and save
    os.makedirs(os.path.dirname(per_video_path), exist_ok=True)
    
    with open(per_video_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['video_name', 'mean', 'std', 'min', 'max'])
        
        for video, fold_stats in sorted(video_fold_data.items()):
            # Average across folds for this video
            video_stats = {k: np.mean([fs[k] for fs in fold_stats]) for k in fold_stats[0]}
            
            writer.writerow([
                video,
                f"{video_stats['mean']:.6f}",
                f"{video_stats['std']:.6f}",
                f"{video_stats['min']:.6f}",
                f"{video_stats['max']:.6f}"
            ])
    
    print(f"✅ Per-video metrics saved: {per_video_path}")


def save_attention_csv(csv_path, video_name, fold, top_indices, top_scores, all_scores, mode='a'):
    """
    Save attention metrics to CSV file.
    
    Args:
        csv_path: Path to CSV file
        video_name: Name of the video file
        fold: Fold number (or None)
        top_indices: Indices of top-3 frames
        top_scores: Scores of top-3 frames
        all_scores: All attention scores for the video
        mode: File mode ('w' for write/overwrite, 'a' for append)
    """
    # Create output directory if needed
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    
    # Check if file exists and is empty to write header
    file_exists = os.path.exists(csv_path) and os.path.getsize(csv_path) > 0
    
    with open(csv_path, mode, newline='') as f:
        writer = csv.writer(f)
        
        # Write header if file doesn't exist or is empty
        if not file_exists or mode == 'w':
            writer.writerow([
                'video_name', 'fold', 'total_frames',
                'top1_frame', 'top1_score',
                'top2_frame', 'top2_score',
                'top3_frame', 'top3_score',
                'mean_attention', 'std_attention',
                'min_attention', 'max_attention'
            ])
        
        # Write data row
        fold_str = f"fold{fold}" if fold is not None else "no_fold"
        writer.writerow([
            video_name,
            fold_str,
            len(all_scores),
            top_indices[0], f"{top_scores[0]:.6f}",
            top_indices[1], f"{top_scores[1]:.6f}",
            top_indices[2], f"{top_scores[2]:.6f}",
            f"{all_scores.mean():.6f}",
            f"{all_scores.std():.6f}",
            f"{all_scores.min():.6f}",
            f"{all_scores.max():.6f}"
        ])
    
    print(f"✅ CSV updated: {csv_path}")


def load_model(checkpoint_path, device):
    """Load a model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    from types import SimpleNamespace
    config = SimpleNamespace(**checkpoint['config'])
    model_type = getattr(config, 'model_type', 'attention_pool')
    model = create_ablation_model(model_type, config).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model


def batch_process_video(video_path, model_base_path, folds, output_dir, csv_path, device):
    """
    Efficiently process one video across multiple folds.
    Loads the video once and processes it with each fold's model.
    """
    video_name = Path(video_path).stem
    
    print("\n" + "="*70)
    print(f"BATCH MODE: Processing {video_name}")
    print("="*70)
    
    # Load video once
    print("Loading video...")
    frames = load_and_preprocess_video(video_path)
    print(f"  ✓ {len(frames)} frames loaded")
    
    # Process with each fold
    for fold in folds:
        checkpoint_path = f"{model_base_path}/fold{fold}/checkpoint_best.pth"
        
        if not os.path.exists(checkpoint_path):
            print(f"\n⚠️  Skipping fold{fold} - checkpoint not found: {checkpoint_path}")
            continue
        
        print(f"\nProcessing fold{fold}...")
        
        # Load model
        model = load_model(checkpoint_path, device)
        
        # Get attention scores
        top_indices, top_scores, all_scores = get_top_k_frames(model, frames, device, k=3)
        
        # Print results
        print(f"  Top-3 frames: {top_indices.tolist()}")
        print(f"  Scores: [{top_scores[0]:.4f}, {top_scores[1]:.4f}, {top_scores[2]:.4f}]")
        print(f"  Mean attention: {all_scores.mean():.4f}")
        print(f"  Std attention: {all_scores.std():.4f}")
        print(f"  Min attention: {all_scores.min():.4f}")
        print(f"  Max attention: {all_scores.max():.4f}")
        
        # Save visualization
        output_path = os.path.join(output_dir, f"{video_name}_fold{fold}.png")
        visualize_top_3_frames(frames, top_indices, top_scores, output_path)
        print(f"  ✓ Saved: {output_path}")
        
        # Save to CSV
        save_attention_csv(csv_path, video_name, fold, top_indices, top_scores, all_scores)
        
        # Clean up model to free memory
        del model
        torch.cuda.empty_cache()
    
    print(f"\n✅ Completed all folds for {video_name}")
    
    # Compute and display average metrics
    print("\n" + "="*70)
    print("INTER-VIDEO STATISTICS (averaged across videos):")
    print("="*70)
    metrics = compute_average_metrics(csv_path)
    if metrics:
        print(f"Number of videos: {metrics['num_videos']}\n")
        
        stats = metrics['inter_video_stats']
        print(f"Intra-video mean:  {stats['mean']['mean']:.6f} ± {stats['mean']['std']:.6f}  [{stats['mean']['min']:.6f}, {stats['mean']['max']:.6f}]")
        print(f"Intra-video std:   {stats['std']['mean']:.6f} ± {stats['std']['std']:.6f}  [{stats['std']['min']:.6f}, {stats['std']['max']:.6f}]")
        print(f"Intra-video min:   {stats['min']['mean']:.6f} ± {stats['min']['std']:.6f}  [{stats['min']['min']:.6f}, {stats['min']['max']:.6f}]")
        print(f"Intra-video max:   {stats['max']['mean']:.6f} ± {stats['max']['std']:.6f}  [{stats['max']['min']:.6f}, {stats['max']['max']:.6f}]")
        
        # Save summary to file
        summary_path = csv_path.replace('.csv', '_summary.csv')
        save_summary_metrics(csv_path, summary_path)
    else:
        print("No metrics available yet.")
    print("="*70)


def main():
    parser = argparse.ArgumentParser(description='Visualize top-3 attention frames')
    parser.add_argument('--video', type=str, 
                       default='/capstor/scratch/cscs/mbarbiere/ultr-ai/LusBeninVideos/25-731_QPIG_15_1.mp4',
                       help='Path to video file (default: first video found)')
    parser.add_argument('--checkpoint', type=str,
                       default='/capstor/store/cscs/swissai/a127/ultr-ai/ablation_results/attention_pool_extra3_full_train2/fold3/checkpoint_best.pth',
                       help='Path to model checkpoint (ignored in batch mode)')
    parser.add_argument('--output', type=str, default='visualization_outputs/top3_attention_frames.png',
                       help='Output filename (or directory in batch mode)')
    parser.add_argument('--csv', type=str, default='visualization_outputs/attention_metrics.csv',
                       help='CSV file for attention metrics (will append if exists)')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--fold', type=int, default=None,
                       help='Fold number to append to output filename (optional, ignored in batch mode)')
    
    # Batch mode arguments
    parser.add_argument('--batch-mode', action='store_true',
                       help='Enable batch mode: process one video across all folds efficiently')
    parser.add_argument('--model-base-path', type=str,
                       help='Base path to model checkpoints (e.g., .../attention_pool_extra4), required for batch mode')
    parser.add_argument('--folds', type=int, nargs='+', default=[0, 1, 2, 3, 4],
                       help='List of fold numbers to process in batch mode (default: 0 1 2 3 4)')

    
    args = parser.parse_args()
    
    if not os.path.exists(args.video):
        print(f"❌ Video not found: {args.video}")
        return
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    # BATCH MODE: Process one video across all folds
    if args.batch_mode:
        if not args.model_base_path:
            print("❌ Error: --model-base-path is required for batch mode")
            return
        
        output_dir = args.output if os.path.isdir(args.output) or not args.output.endswith('.png') else os.path.dirname(args.output)
        os.makedirs(output_dir, exist_ok=True)
        
        batch_process_video(
            video_path=args.video,
            model_base_path=args.model_base_path,
            folds=args.folds,
            output_dir=output_dir,
            csv_path=args.csv,
            device=device
        )
        return
    
    # SINGLE MODE: Original behavior
    # Append fold number to output path if provided
    output_path = args.output
    if args.fold is not None:
        path = Path(output_path)
        output_path = str(path.parent / f"{path.stem}_fold{args.fold}{path.suffix}")
    
    print("="*70)
    print(f"Video: {Path(args.video).name}")
    if args.fold is not None:
        print(f"Fold: {args.fold}")
    print(f"NOTE: Model uses Gumbel-Softmax during training, torch.topk during inference")
    
    # Load video
    print("Loading video...")
    frames = load_and_preprocess_video(args.video)
    print(f"  {len(frames)} frames loaded")
    
    # Load model
    print("Loading model...")
    model = load_model(args.checkpoint, device)
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
    
    # Save to CSV
    print(f"\nSaving metrics to CSV...")
    video_name = Path(args.video).stem
    save_attention_csv(args.csv, video_name, args.fold, top_indices, top_scores, all_scores)
    
    # Visualize
    print(f"\nCreating visualization...")
    visualize_top_3_frames(frames, top_indices, top_scores, output_path)
    
    # Compute and display aggregate metrics
    print("\n" + "="*70)
    print("INTER-VIDEO STATISTICS (averaged across videos):")
    print("="*70)
    metrics = compute_average_metrics(args.csv)
    if metrics:
        print(f"Number of videos: {metrics['num_videos']}\n")
        
        stats = metrics['inter_video_stats']
        print(f"Intra-video mean:  {stats['mean']['mean']:.6f} ± {stats['mean']['std']:.6f}  [{stats['mean']['min']:.6f}, {stats['mean']['max']:.6f}]")
        print(f"Intra-video std:   {stats['std']['mean']:.6f} ± {stats['std']['std']:.6f}  [{stats['std']['min']:.6f}, {stats['std']['max']:.6f}]")
        print(f"Intra-video min:   {stats['min']['mean']:.6f} ± {stats['min']['std']:.6f}  [{stats['min']['min']:.6f}, {stats['min']['max']:.6f}]")
        print(f"Intra-video max:   {stats['max']['mean']:.6f} ± {stats['max']['std']:.6f}  [{stats['max']['min']:.6f}, {stats['max']['max']:.6f}]")
        
        # Save summary to file
        summary_path = args.csv.replace('.csv', '_summary.csv')
        save_summary_metrics(args.csv, summary_path)
    else:
        print("No metrics available yet.")
    
    print("="*70)
    print("✅ DONE!")
    

if __name__ == '__main__':
    main()
