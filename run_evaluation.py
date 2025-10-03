import argparse
from evaluate import *


def main():
    # Evaluate the models
    logger.info("Starting Multi-Task Model Comprehensive Evaluation")
    logger.info("=" * 60)

    parser = argparse.ArgumentParser(description='Training for TB detection using ablation models.')

    # Add the experiment name and fold as optional arguments. 
    # Experiment name options are: 3dcnn, attention_pool, cnnlstm, mean_pool, singletask, uniform, vivit
    parser.add_argument('--experiment_name', type=str, help='Name of the experiment')

    # Parse the arguments
    args = parser.parse_args()

    # Check that experiment_name is provided
    if not args.experiment_name:
        raise ValueError("Please provide an experiment name using --experiment_name")
    
    # Log the experiment name
    logger.info(f"Experiment Name: {args.experiment_name}")
    
    # Configuration paths for all folds - UPDATE THESE TO MATCH YOUR ACTUAL CONFIG FILES
    config_paths = [
        f"./ablation_results/{args.experiment_name}/fold0/config.yaml",
        f"./ablation_results/{args.experiment_name}/fold1/config.yaml",
        f"./ablation_results/{args.experiment_name}/fold2/config.yaml",
        f"./ablation_results/{args.experiment_name}/fold3/config.yaml",
        f"./ablation_results/{args.experiment_name}/fold4/config.yaml",
    ]
    
    # Paths to checkpoins
    model_paths = [
        f"./ablation_results/{args.experiment_name}/fold0/checkpoints/checkpoint_best.pth",
        f"./ablation_results/{args.experiment_name}/fold1/checkpoints/checkpoint_best.pth",
        f"./ablation_results/{args.experiment_name}/fold2/checkpoints/checkpoint_best.pth",
        f"./ablation_results/{args.experiment_name}/fold3/checkpoints/checkpoint_best.pth",
        f"./ablation_results/{args.experiment_name}/fold4/checkpoints/checkpoint_best.pth",
    ]
    
    logger.info(f"Found {len(config_paths)} folds to process")
    
    # First, check which files exist
    valid_folds = []
    for i in range(len(config_paths)):
        config_exists = os.path.exists(config_paths[i])
        
        # If model_path is None, we'll check config file for model_path
        if model_paths[i] is None:
            model_exists = True  # Will be validated when loading config
        else:
            model_exists = os.path.exists(model_paths[i])
        
        logger.info(f"\nFold {i}:")
        logger.info(f"  Config: {'FOUND' if config_exists else 'MISSING'} {config_paths[i]}")
        if model_paths[i] is not None:
            logger.info(f"  Model:  {'FOUND' if model_exists else 'MISSING'} {model_paths[i]}")
        else:
            logger.info(f"  Model:  Will use path from config file")
        
        if config_exists and model_exists:
            valid_folds.append(i)
        else:
            logger.warning(f"  Status: Skipping fold {i} (missing files)")
    
    logger.info(f"Found {len(valid_folds)} valid folds: {valid_folds}")
    
    if not valid_folds:
        logger.error("No valid folds found. Please check the file paths.")
        exit()
    
    # Process each valid fold
    successful_folds = []
    failed_folds = []
    
    for i in valid_folds:
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing Fold {i}")
        logger.info(f"{'='*60}")
        
        test_config = {
            'config_path': config_paths[i],
            'model_path': model_paths[i],  # Can be None if specified in config
            'split': 'all',
            'fold': i,
            'output_dir': f'ablation_results/{args.experiment_name}/eval_results',
            'gpu_id': 0,
            'save_complex_data': True,
            'batch_size_override': None,
        }
        
        try:
            logger.info(f"Starting evaluation for fold {i}...")
            patient_df, site_df, complex_data, metrics, saved_files = run_comprehensive_evaluation(test_config)
            
            logger.info(f"Fold {i} completed successfully!")
            logger.info(f"   - Patients evaluated: {len(patient_df)}")
            logger.info(f"   - Sites evaluated: {len(site_df)}")
            logger.info(f"   - Files saved: {len(saved_files)}")
            
            # Print key metrics if available
            if metrics:
                for task_name in ['TB Label', 'Pneumonia Label', 'Covid Label']:
                    task_metrics = {k.replace(f'{task_name}_', ''): v for k, v in metrics.items() 
                                  if k.startswith(f'{task_name}_')}
                    if task_metrics:
                        auc = task_metrics.get('auc', 'N/A')
                        acc = task_metrics.get('accuracy', 'N/A')
                        logger.info(f"   - {task_name}: AUC={auc:.4f if isinstance(auc, (int, float)) else auc}, ACC={acc:.4f if isinstance(acc, (int, float)) else acc}")
            
            successful_folds.append(i)
            
        except Exception as e:
            logger.error(f"Error processing fold {i}: {e}")
            failed_folds.append(i)
            
            # Print traceback for debugging
            import traceback
            logger.error("Full error traceback:")
            traceback.print_exc()
            
            # Continue with next fold
            continue
    
    # Final summary
    logger.info(f"\n{'='*60}")
    logger.info(f"EVALUATION SUMMARY")
    logger.info(f"{'='*60}")
    logger.info(f"Successful folds: {successful_folds} ({len(successful_folds)}/{len(valid_folds)})")
    if failed_folds:
        logger.warning(f"Failed folds: {failed_folds}")

    logger.info(f"\nResults saved to: ablation_results/{args.experiment_name}/eval_results")

    if successful_folds:
        logger.info(f"Evaluation completed for {len(successful_folds)} folds!")
        logger.info("Check the results directory for detailed outputs:")
        logger.info("  - CSV files: patient and site-level predictions")
        logger.info("  - HDF5 files: complex model outputs and features")
        logger.info("  - JSON files: evaluation metrics")
    else:
        logger.error("No folds completed successfully. Please check the errors above.")

    # Optional: Create a consolidated summary across all folds
    if len(successful_folds) > 1:
        logger.info(f"\nCreating consolidated summary across {len(successful_folds)} folds...")
        try:
            create_cross_fold_summary(successful_folds, f'ablation_results/{args.experiment_name}/eval_results')
        except Exception as e:
            logger.warning(f"Could not create cross-fold summary: {e}")



if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.exception(f"Error in evaluation: {e}")
        raise