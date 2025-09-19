# scripts/analyze_results.py
import os
import sys
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import json
from collections import defaultdict

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from scripts.evaluate import load_evaluation_results

# Set up plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


def analyze_tb_performance(tabular_df):
    """Analyze TB classification performance."""
    patient_data = tabular_df[tabular_df['record_type'] == 'patient'].copy()
    
    print("TB Classification Analysis:")
    print("=" * 40)
    
    # Basic statistics
    total_patients = len(patient_data)
    tb_positive = (patient_data['tb_label'] == 1).sum()
    tb_negative = (patient_data['tb_label'] == 0).sum()
    
    print(f"Total patients: {total_patients}")
    print(f"TB positive: {tb_positive} ({tb_positive/total_patients*100:.1f}%)")
    print(f"TB negative: {tb_negative} ({tb_negative/total_patients*100:.1f}%)")
    
    # Prediction accuracy
    correct_predictions = (patient_data['tb_pred'] == patient_data['tb_label']).sum()
    accuracy = correct_predictions / total_patients
    print(f"Accuracy: {accuracy:.4f}")
    
    # Confidence analysis
    high_confidence = (patient_data['tb_prob'] > 0.8) | (patient_data['tb_prob'] < 0.2)
    print(f"High confidence predictions: {high_confidence.sum()} ({high_confidence.sum()/total_patients*100:.1f}%)")
    
    return patient_data


def analyze_site_patterns(tabular_df):
    """Analyze site-level patterns."""
    site_data = tabular_df[tabular_df['record_type'] == 'site'].copy()
    
    print("\nSite Pattern Analysis:")
    print("=" * 40)
    
    # Site distribution
    site_counts = site_data['site_index'].value_counts().sort_index()
    print("Site distribution:")
    for site_idx, count in site_counts.items():
        print(f"  Site {site_idx}: {count} occurrences")
    
    # Sites per patient
    sites_per_patient = site_data.groupby('patient_id').size()
    print(f"\nSites per patient:")
    print(f"  Mean: {sites_per_patient.mean():.2f}")
    print(f"  Median: {sites_per_patient.median():.1f}")
    print(f"  Min: {sites_per_patient.min()}")
    print(f"  Max: {sites_per_patient.max()}")
    
    # Site-level TB correlation
    print(f"\nSite-level TB correlation:")
    for site_idx in sorted(site_data['site_index'].unique()):
        site_subset = site_data[site_data['site_index'] == site_idx]
        tb_rate = site_subset['tb_label'].mean()
        print(f"  Site {site_idx}: {len(site_subset)} samples, TB rate: {tb_rate:.3f}")
    
    return site_data


def analyze_pathology_predictions(tabular_df):
    """Analyze pathology prediction patterns."""
    site_data = tabular_df[tabular_df['record_type'] == 'site'].copy()
    
    print("\nPathology Analysis:")
    print("=" * 40)
    
    pathology_names = ['a_lines', 'b_lines', 'small_consolidation', 
                      'large_consolidation', 'pleural_effusion']
    
    pathology_results = {}
    
    for pathology in pathology_names:
        finding_col = f'site_finding_{pathology}'
        pred_col = f'site_pathology_{pathology}_pred'
        prob_col = f'site_pathology_{pathology}_prob'
        
        if finding_col in site_data.columns and pred_col in site_data.columns:
            # Filter valid labels (not -1)
            valid_mask = site_data[finding_col] >= 0
            valid_data = site_data[valid_mask]
            
            if len(valid_data) > 0:
                true_labels = valid_data[finding_col]
                predictions = valid_data[pred_col]
                probabilities = valid_data[prob_col] if prob_col in valid_data.columns else None
                
                accuracy = (true_labels == predictions).mean()
                positive_rate = true_labels.mean()
                
                # Calculate additional metrics
                from sklearn.metrics import precision_score, recall_score, f1_score
                precision = precision_score(true_labels, predictions, zero_division=0)
                recall = recall_score(true_labels, predictions, zero_division=0)
                f1 = f1_score(true_labels, predictions, zero_division=0)
                
                pathology_results[pathology] = {
                    'samples': len(valid_data),
                    'positive_rate': positive_rate,
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1': f1
                }
                
                print(f"{pathology.replace('_', ' ').title()}:")
                print(f"  Valid samples: {len(valid_data)}")
                print(f"  Positive rate: {positive_rate:.3f}")
                print(f"  Accuracy: {accuracy:.3f}")
                print(f"  Precision: {precision:.3f}")
                print(f"  Recall: {recall:.3f}")
                print(f"  F1-score: {f1:.3f}")
    
    return pathology_results


def analyze_attention_patterns(tabular_df, complex_data):
    """Analyze MIL attention patterns."""
    if not complex_data['mil_attention']:
        print("No attention data available for analysis")
        return
    
    print("\nAttention Pattern Analysis:")
    print("=" * 40)
    
    patient_data = tabular_df[tabular_df['record_type'] == 'patient'].copy()
    
    # Analyze attention distributions
    attention_stats = []
    
    for patient_id in patient_data['patient_id']:
        if patient_id in complex_data['mil_attention']:
            attention = complex_data['mil_attention'][patient_id]
            tb_label = patient_data[patient_data['patient_id'] == patient_id]['tb_label'].iloc[0]
            
            attention_stats.append({
                'patient_id': patient_id,
                'tb_label': tb_label,
                'attention_max': attention.max(),
                'attention_min': attention.min(),
                'attention_std': attention.std(),
                'attention_entropy': -np.sum(attention * np.log(attention + 1e-8))
            })
    
    attention_df = pd.DataFrame(attention_stats)
    
    print(f"Attention statistics for {len(attention_df)} patients:")
    print("TB Positive vs Negative attention patterns:")
    
    for metric in ['attention_max', 'attention_std', 'attention_entropy']:
        tb_pos = attention_df[attention_df['tb_label'] == 1][metric]
        tb_neg = attention_df[attention_df['tb_label'] == 0][metric]
        
        print(f"{metric}:")
        print(f"  TB+: {tb_pos.mean():.4f} ± {tb_pos.std():.4f}")
        print(f"  TB-: {tb_neg.mean():.4f} ± {tb_neg.std():.4f}")
    
    return attention_df


def create_site_visualizations(tabular_df, complex_data, output_dir, filename_base):
    """Create comprehensive site-level visualizations."""
    site_data = tabular_df[tabular_df['record_type'] == 'site'].copy()
    
    print("\nCreating site-level visualizations...")
    
    # 1. Site distribution visualization
    plt.figure(figsize=(12, 8))
    
    # Site count distribution
    plt.subplot(2, 2, 1)
    site_counts = site_data['site_index'].value_counts().sort_index()
    plt.bar(site_counts.index, site_counts.values)
    plt.xlabel('Site Index')
    plt.ylabel('Count')
    plt.title('Distribution of Sites')
    plt.xticks(rotation=45)
    
    # TB rate by site
    plt.subplot(2, 2, 2)
    site_tb_rates = site_data.groupby('site_index')['tb_label'].agg(['mean', 'count'])
    plt.bar(site_tb_rates.index, site_tb_rates['mean'])
    plt.xlabel('Site Index')
    plt.ylabel('TB Positive Rate')
    plt.title('TB Positive Rate by Site')
    plt.xticks(rotation=45)
    
    # Sites per patient distribution
    plt.subplot(2, 2, 3)
    sites_per_patient = site_data.groupby('patient_id').size()
    plt.hist(sites_per_patient.values, bins=range(1, max(sites_per_patient.values) + 2), alpha=0.7)
    plt.xlabel('Number of Sites per Patient')
    plt.ylabel('Frequency')
    plt.title('Distribution of Sites per Patient')
    
    # Site position distribution
    plt.subplot(2, 2, 4)
    plt.hist(site_data['site_position'].values, bins=range(0, max(site_data['site_position'].values) + 2), alpha=0.7)
    plt.xlabel('Site Position')
    plt.ylabel('Frequency')
    plt.title('Distribution of Site Positions')
    
    plt.tight_layout()
    site_dist_path = os.path.join(output_dir, f'{filename_base}_site_distribution.png')
    plt.savefig(site_dist_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Site distribution plot saved to {site_dist_path}")
    
    # 2. Pathology prediction visualization
    pathology_names = ['a_lines', 'b_lines', 'small_consolidation', 
                      'large_consolidation', 'pleural_effusion']
    
    available_pathologies = []
    for pathology in pathology_names:
        finding_col = f'site_finding_{pathology}'
        pred_col = f'site_pathology_{pathology}_pred'
        if finding_col in site_data.columns and pred_col in site_data.columns:
            available_pathologies.append(pathology)
    
    if available_pathologies:
        n_pathologies = len(available_pathologies)
        fig, axes = plt.subplots(2, (n_pathologies + 1) // 2, figsize=(15, 10))
        if n_pathologies == 1:
            axes = [axes]
        elif n_pathologies <= 2:
            axes = axes.flatten()
        else:
            axes = axes.flatten()
        
        for i, pathology in enumerate(available_pathologies):
            finding_col = f'site_finding_{pathology}'
            pred_col = f'site_pathology_{pathology}_pred'
            
            # Filter valid labels
            valid_mask = site_data[finding_col] >= 0
            valid_data = site_data[valid_mask]
            
            if len(valid_data) > 0:
                # Confusion matrix
                from sklearn.metrics import confusion_matrix
                cm = confusion_matrix(valid_data[finding_col], valid_data[pred_col])
                
                if i < len(axes):
                    sns.heatmap(cm, annot=True, fmt='d', ax=axes[i], 
                              xticklabels=['Negative', 'Positive'],
                              yticklabels=['Negative', 'Positive'])
                    axes[i].set_title(f'{pathology.replace("_", " ").title()}\nConfusion Matrix')
                    axes[i].set_xlabel('Predicted')
                    axes[i].set_ylabel('Actual')
        
        # Hide unused subplots
        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)
        
        plt.tight_layout()
        pathology_path = os.path.join(output_dir, f'{filename_base}_pathology_confusion_matrices.png')
        plt.savefig(pathology_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Pathology confusion matrices saved to {pathology_path}")
    
    # 3. Site features visualization (if available)
    if complex_data['site_features']:
        print("Creating site features visualization...")
        
        # Collect site features
        site_features = []
        site_labels = []
        site_indices = []
        
        for (patient_id, site_idx), features in complex_data['site_features'].items():
            # Find corresponding site data
            site_match = site_data[
                (site_data['patient_id'] == patient_id) & 
                (site_data['site_index'] == site_idx)
            ]
            
            if len(site_match) > 0:
                site_features.append(features)
                site_labels.append(site_match['tb_label'].iloc[0])
                site_indices.append(site_idx)
        
        if len(site_features) > 30:  # Need enough samples for t-SNE
            site_features = np.array(site_features)
            site_labels = np.array(site_labels)
            site_indices = np.array(site_indices)
            
            # t-SNE visualization
            tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(site_features)//2))
            site_features_tsne = tsne.fit_transform(site_features)
            
            plt.figure(figsize=(15, 5))
            
            # By TB label
            plt.subplot(1, 3, 1)
            colors = ['red' if label == 1 else 'blue' for label in site_labels]
            plt.scatter(site_features_tsne[:, 0], site_features_tsne[:, 1], 
                       c=colors, alpha=0.6, s=30)
            plt.title('Site Features by TB Label')
            plt.xlabel('t-SNE 1')
            plt.ylabel('t-SNE 2')
            plt.legend(['TB-', 'TB+'])
            
            # By site index
            plt.subplot(1, 3, 2)
            unique_sites = np.unique(site_indices)
            colors = plt.cm.tab10(np.linspace(0, 1, len(unique_sites)))
            for i, site_idx in enumerate(unique_sites):
                mask = site_indices == site_idx
                plt.scatter(site_features_tsne[mask, 0], site_features_tsne[mask, 1], 
                           c=[colors[i]], alpha=0.6, s=30, label=f'Site {site_idx}')
            plt.title('Site Features by Site Index')
            plt.xlabel('t-SNE 1')
            plt.ylabel('t-SNE 2')
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            
            # Feature importance (if possible)
            plt.subplot(1, 3, 3)
            feature_std = site_features.std(axis=0)
            top_features = np.argsort(feature_std)[-20:]  # Top 20 most variable features
            plt.barh(range(len(top_features)), feature_std[top_features])
            plt.xlabel('Standard Deviation')
            plt.ylabel('Feature Index')
            plt.title('Top Variable Features')
            
            plt.tight_layout()
            site_features_path = os.path.join(output_dir, f'{filename_base}_site_features_tsne.png')
            plt.savefig(site_features_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Site features visualization saved to {site_features_path}")
    
    # 4. Attention heatmap (if available)
    if complex_data['mil_attention']:
        print("Creating attention heatmap...")
        
        # Collect attention data
        patient_data = tabular_df[tabular_df['record_type'] == 'patient'].copy()
        attention_data = []
        
        for _, row in patient_data.iterrows():
            patient_id = row['patient_id']
            if patient_id in complex_data['mil_attention']:
                attention = complex_data['mil_attention'][patient_id]
                tb_label = row['tb_label']
                
                # Pad attention to fixed length for visualization
                max_sites = 15  # Assuming max 15 sites
                padded_attention = np.zeros(max_sites)
                padded_attention[:len(attention)] = attention
                
                attention_data.append({
                    'patient_id': patient_id,
                    'tb_label': tb_label,
                    'attention': padded_attention
                })
        
        if attention_data:
            # Sort by TB label for better visualization
            attention_data.sort(key=lambda x: x['tb_label'])
            
            # Create attention matrix
            attention_matrix = np.array([item['attention'] for item in attention_data])
            tb_labels = [item['tb_label'] for item in attention_data]
            
            plt.figure(figsize=(12, 8))
            
            # Create heatmap
            im = plt.imshow(attention_matrix, cmap='viridis', aspect='auto')
            plt.colorbar(im, label='Attention Weight')
            plt.xlabel('Site Position')
            plt.ylabel('Patient (sorted by TB label)')
            plt.title('MIL Attention Heatmap')
            
            # Add TB label indicators
            tb_change_idx = np.where(np.diff(tb_labels))[0]
            for idx in tb_change_idx:
                plt.axhline(y=idx + 0.5, color='red', linestyle='--', alpha=0.7)
            
            # Add labels
            plt.text(0.02, 0.02, 'TB-', transform=plt.gca().transAxes, 
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="blue", alpha=0.7))
            if len(tb_change_idx) > 0:
                plt.text(0.02, 0.98, 'TB+', transform=plt.gca().transAxes, 
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="red", alpha=0.7))
            
            attention_heatmap_path = os.path.join(output_dir, f'{filename_base}_attention_heatmap.png')
            plt.savefig(attention_heatmap_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Attention heatmap saved to {attention_heatmap_path}")


def visualize_feature_embeddings(complex_data, tabular_df, output_dir, filename_base):
    """Create visualizations of patient and site embeddings."""
    if not complex_data['patient_features']:
        print("No patient features available for visualization")
        return
    
    print("\nCreating feature visualizations...")
    
    # Prepare patient feature data
    patient_data = tabular_df[tabular_df['record_type'] == 'patient'].copy()
    
    patient_features = []
    patient_labels = []
    patient_ids = []
    
    for _, row in patient_data.iterrows():
        patient_id = row['patient_id']
        if patient_id in complex_data['patient_features']:
            patient_features.append(complex_data['patient_features'][patient_id])
            patient_labels.append(row['tb_label'])
            patient_ids.append(patient_id)
    
    if len(patient_features) == 0:
        print("No matching patient features found")
        return
    
    patient_features = np.array(patient_features)
    patient_labels = np.array(patient_labels)
    
    # PCA visualization
    if patient_features.shape[1] > 2:
        pca = PCA(n_components=2)
        patient_features_2d = pca.fit_transform(patient_features)
        
        plt.figure(figsize=(10, 8))
        colors = ['red' if label == 1 else 'blue' for label in patient_labels]
        plt.scatter(patient_features_2d[:, 0], patient_features_2d[:, 1], 
                   c=colors, alpha=0.6, s=50)
        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2f})')
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2f})')
        plt.title('Patient Features - PCA Visualization')
        plt.legend(['TB-', 'TB+'])
        plt.grid(True, alpha=0.3)
        
        pca_path = os.path.join(output_dir, f'{filename_base}_patient_features_pca.png')
        plt.savefig(pca_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"PCA visualization saved to {pca_path}")
    
    # t-SNE visualization (if enough samples)
    if len(patient_features) > 30:
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(patient_features)//2))
        patient_features_tsne = tsne.fit_transform(patient_features)
        
        plt.figure(figsize=(10, 8))
        colors = ['red' if label == 1 else 'blue' for label in patient_labels]
        plt.scatter(patient_features_tsne[:, 0], patient_features_tsne[:, 1], 
                   c=colors, alpha=0.6, s=50)
        plt.xlabel('t-SNE 1')
        plt.ylabel('t-SNE 2')
        plt.title('Patient Features - t-SNE Visualization')
        plt.legend(['TB-', 'TB+'])
        plt.grid(True, alpha=0.3)
        
        tsne_path = os.path.join(output_dir, f'{filename_base}_patient_features_tsne.png')
        plt.savefig(tsne_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"t-SNE visualization saved to {tsne_path}")


def create_comprehensive_report(tabular_df, complex_data, metrics, output_dir, filename_base, 
                              experiment_name, split_name, fold_num):
    """Create a comprehensive analysis report."""
    report_path = os.path.join(output_dir, f'{filename_base}_analysis_report.txt')
    
    with open(report_path, 'w') as f:
        f.write("TB-DRL-MIL Evaluation Analysis Report\n")
        f.write("=" * 50 + "\n\n")
        
        # Experiment details
        f.write(f"Experiment: {experiment_name}\n")
        f.write(f"Split: {split_name}\n")
        if fold_num is not None:
            f.write(f"Fold: {fold_num}\n")
        f.write(f"Analysis date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Dataset overview
        patient_data = tabular_df[tabular_df['record_type'] == 'patient']
        site_data = tabular_df[tabular_df['record_type'] == 'site']
        
        f.write(f"Dataset Overview:\n")
        f.write(f"Total patients: {len(patient_data)}\n")
        f.write(f"Total sites: {len(site_data)}\n")
        f.write(f"TB positive patients: {(patient_data['tb_label'] == 1).sum()}\n")
        f.write(f"TB negative patients: {(patient_data['tb_label'] == 0).sum()}\n")
        f.write(f"Average sites per patient: {len(site_data) / len(patient_data):.2f}\n\n")
        
        # Performance metrics
        f.write(f"Performance Metrics:\n")
        for metric, value in metrics.items():
            if isinstance(value, (int, float)):
                f.write(f"{metric}: {value:.4f}\n")
        f.write("\n")
        
        # Site distribution
        f.write(f"Site Distribution:\n")
        site_counts = site_data['site_index'].value_counts().sort_index()
        for site_idx, count in site_counts.items():
            f.write(f"Site {site_idx}: {count} occurrences\n")
        f.write("\n")
        
        # Data completeness
        if complex_data:
            f.write(f"Data Completeness:\n")
            f.write(f"Patient features: {len(complex_data['patient_features'])} patients\n")
            f.write(f"Site features: {len(complex_data['site_features'])} sites\n")
            f.write(f"MIL attention: {len(complex_data['mil_attention'])} patients\n")
            f.write(f"RL data: {len(complex_data['site_rl_data'])} sites\n")
            f.write(f"Pathology scores: {len(complex_data['pathology_scores_full'])} sites\n")
    
    print(f"Comprehensive report saved to {report_path}")


def main():
    parser = argparse.ArgumentParser(description='Analyze detailed evaluation results with fold support')
    parser.add_argument('--results_dir', type=str, required=True,
                       help='Directory containing evaluation results')
    parser.add_argument('--split', type=str, default='test',
                       help='Dataset split to analyze')
    parser.add_argument('--experiment_name', type=str, required=True,
                       help='Experiment name used during evaluation')
    parser.add_argument('--fold', type=int, help='Fold number (if applicable)')
    parser.add_argument('--output_dir', type=str, help='Output directory for analysis (default: same as results_dir)')
    
    args = parser.parse_args()
    
    if args.output_dir is None:
        args.output_dir = args.results_dir
    
    # Create filename base
    if args.fold is not None:
        filename_base = f'{args.split}_{args.experiment_name}_fold{args.fold}'
    else:
        filename_base = f'{args.split}_{args.experiment_name}'
    
    # Load evaluation results
    print(f"Loading evaluation results from {args.results_dir}")
    print(f"Looking for: {filename_base}")
    
    try:
        tabular_df, complex_data, metrics = load_evaluation_results(
            args.results_dir, args.split, args.experiment_name, args.fold
        )
        print(f"Successfully loaded results for {args.split} split")
    except Exception as e:
        print(f"Error loading results: {e}")
        return
    
    # Run analyses
    print("\n" + "="*70)
    print("DETAILED ANALYSIS")
    print("="*70)
    
    # TB performance analysis
    patient_data = analyze_tb_performance(tabular_df)
    
    # Site pattern analysis
    site_data = analyze_site_patterns(tabular_df)
    
    # Pathology analysis
    pathology_results = analyze_pathology_predictions(tabular_df)
    
    # Attention analysis
    attention_df = analyze_attention_patterns(tabular_df, complex_data)
    
    # Create visualizations
    print("\n" + "="*70)
    print("CREATING VISUALIZATIONS")
    print("="*70)
    
    # Site-level visualizations
    create_site_visualizations(tabular_df, complex_data, args.output_dir, filename_base)
    
    # Patient-level feature visualizations
    visualize_feature_embeddings(complex_data, tabular_df, args.output_dir, filename_base)
    
    # Create comprehensive report
    create_comprehensive_report(
        tabular_df, complex_data, metrics, args.output_dir, filename_base,
        args.experiment_name, args.split, args.fold
    )
    
    print(f"\nAnalysis complete. All results saved to {args.output_dir}")
    print(f"Files created with base name: {filename_base}")


if __name__ == "__main__":
    main()