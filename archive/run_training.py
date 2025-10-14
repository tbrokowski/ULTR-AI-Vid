# Import everything from the train_ablation file
from train_ablation_models import *

def main():
    """Main training function with command line argument support."""
    
    # Parse command line arguments and load configuration
    config = parse_args_and_load_config()
    
    logger.info("TB Ablation Classification Configuration:")
    for key, value in config.to_dict().items():
        logger.info(f"  {key}: {value}")
    
    # Create all required directories after configuration is fully loaded
    config.create_directories()
    config_path = os.path.join(config.experiment_dir, "config.yaml")
    config.save(config_path)
    logger.info(f"Configuration saved to {config_path}")
    
    # GPU/Device information
    if torch.cuda.is_available():
        logger.info(f"Using device: {config.device}")
        logger.info(f"Device name: {torch.cuda.get_device_name(0)}")
        logger.info(f"Number of GPUs available: {torch.cuda.device_count()}")
        print_gpu_memory()
    else:
        logger.info("Using CPU")

    # Initialize trainer
    trainer = AblationTrainer(config)
    
    # Check for resume checkpoint
    resume_checkpoint = None
    if hasattr(config, 'resume_from_checkpoint') and config.resume_from_checkpoint is not None:
        if not os.path.exists(config.resume_from_checkpoint):
            logger.error(f"Resume checkpoint not found: {config.resume_from_checkpoint}")
            return
        resume_checkpoint = config.resume_from_checkpoint
        logger.info(f"Will resume training from: {resume_checkpoint}")
    
    # Check if we're in evaluation-only mode
    # if not config.train:
    #     logger.info("Running in evaluation-only mode")
    #     trainer._evaluate_best_model()
    #     return
    
    # Start training
    logger.info(f"Starting TB training with ablation model: {config.model_type}...")
    best_metric, best_epoch = trainer.train(resume_from_checkpoint=resume_checkpoint)
    logger.info(f"Training complete! Best metric: {best_metric:.4f} at epoch {best_epoch+1}")

    return best_metric, best_epoch


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.exception(f"Error in training: {e}")
        raise



# # 3D CNN ablation
# python3 run_training.py --experiment_name 3dcnn --fold 0 --video_folder /capstor/scratch/cscs/mbarbiere/ultr-ai/LusBeninVideos

# # CNN-LSTM ablation
# python3 run_training.py --experiment_name cnnlstm --fold 0 --video_folder /capstor/scratch/cscs/mbarbiere/ultr-ai/LusBeninVideos

# # Video Transformer (ViViT) ablation
# python3 run_training.py --experiment_name vivit --fold 0 --video_folder /capstor/scratch/cscs/mbarbiere/ultr-ai/LusBeninVideos

# # Attention pooling ablation
# python3 run_training.py --experiment_name attention_pool --fold 0 --video_folder /capstor/scratch/cscs/mbarbiere/ultr-ai/LusBeninVideos

# # Mean pooling ablation
# python3 run_training.py --experiment_name mean_pool --fold 4 --video_folder /capstor/scratch/cscs/mbarbiere/ultr-ai/LusBeninVideos

# # Single task ablation
# python3 run_training.py --experiment_name singletask --fold 0 --video_folder /capstor/scratch/cscs/mbarbiere/ultr-ai/LusBeninVideos

# # Uniform/No-RL ablation
# python3 run_training.py --experiment_name uniform --fold 0 --video_folder /capstor/scratch/cscs/mbarbiere/ultr-ai/LusBeninVideos

