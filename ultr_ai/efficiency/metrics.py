"""
This file contains metrics to evaluate efficiency when performing inference of the model.
"""

class Config:
    def __init__(self, params=None):
        """Initialize configuration with defaults and optional overrides."""
        self._set_defaults()
        
        if params:
            for key, value in params.items():
                setattr(self, key, value)
    
    def _set_defaults(self):
        """Set default configuration values."""
        # Training mode
        self.train = True
        
        # Data paths
        self.root_dir = ""
        self.labels_csv = ""
        self.file_metadata_csv = ""
        self.split_csv = ''
        self.video_folder = 'videos'
        self.image_folder = 'images'
        
        # Model config
        self.model_type = 'no_rl'
        self.model_name = 'ablation_tb_classifier_fold0'
        self.backbone = 'resnet18'
        self.freeze_backbone = False
        self.hidden_dim = 512
        self.dropout_rate = 0.3
        self.num_pathologies = 4
        self.pretrained = True
        self.num_classes = 1
        self.in_channels = 3
        self.reset_optimizers = False
        
        # Data preprocessing
        self.target_height = 224
        self.target_width = 224
        self.depth_filter = '15'
        self.frame_sampling = 32
        self.num_sites = 15
        self.mode = 'video'
        self.pooling = 'attention'
        
        # Training settings
        self.task = "TB Label"
        self.batch_size = 2
        self.num_workers = 6
        self.learning_rate = 0.00001
        self.weight_decay = 0.00001
        self.num_epochs = 20
        self.early_stopping_patience = 8
        self.accumulation_steps = 8
        self.use_amp = True
        self.seed = 42
        
        # Multi-task config
        self.active_tasks = ['TB Label']
        self.use_pathology_loss = True
        self.task_weights = {'TB Label': 1.0}
        
        # Attention settings
        self.attention_temperature = 0.5  # Temperature for soft attention (lower = sharper)
        
        # Dataset parameters
        self.files_per_site = 1
        self.site_order = None
        self.pad_missing_sites = True
        self.max_sites = 15
        
        self.classification_type = "binary"
        self.pos_weight = 1.4
        
        # Evaluation settings
        self.eval_metric = "auc"
        self.eval_metric_goal = "max"
        self.evaluate_best_valid_model = True
        
        self.local_weights_dir = '/NetworkArchitecture/CLIP_weights'
        
        # Optimizer settings
        self.backbone_lr = 0.00001
        self.backbone_weight_decay = 0.00001
        self.backbone_eta_min = 1e-6
        
        self.pathology_lr = 0.0001
        self.pathology_weight_decay = 0.00001
        self.pathology_eta_min = 1e-6
        
        self.patient_pipeline_lr = 0.001
        self.patient_pipeline_weight_decay = 0.00001
        self.patient_pipeline_eta_min = 1e-6
        
        # Directories
        self.log_dir = "logs"
        self.save_dir = "models"
        self.checkpoint_dir = "checkpoints"
        self.pred_save_dir = "predictions"
        self.checkpoint_base_dir = "/capstor/store/cscs/swissai/a127/ultr-ai"
        self.experiment_dir = None  # Will be set based on experiment_name
        
        # Pathology settings
        self.pathology_pos_weights = [1.0, 4.0, 4.0, 4.0]
        self.pathology_classes = [
            'A-line',
            'Large consolidations', 
            'Pleural Effusion',
            'Other Pathology'
        ]
        
        # Distributed training settings
        self.distributed = False
        self.world_size = 1
        self.rank = 0
        self.local_rank = 0
        self.dist_backend = 'nccl'
        self.dist_url = 'env://'
        
        # Device (will be set based on local_rank)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def load_from_yaml(self, yaml_path):
        """Load configuration from YAML file (upstream-friendly)."""
        if not os.path.exists(yaml_path):
            logger.warning(f"Config file not found: {yaml_path}")
        try:
            with open(yaml_path, 'r') as f:
                yaml_config = yaml.safe_load(f) or {}
            logger.info(f"Loading configuration from {yaml_path}")

            # Accept ALL keys (no unknown-key warnings)
            for key, value in yaml_config.items():
                setattr(self, key, value)

            # Ensure experiment_dir is set after potential overrides
            if not getattr(self, 'experiment_dir', None):
                self.experiment_dir = os.path.join(self.checkpoint_dir, self.model_name)

            # Normalize output directories to external /capstor location
            CAPSTOR_ROOT = os.environ.get(
                "CAPSTOR_ROOT",
                "/capstor/store/cscs/swissai/a127/ultr-ai"
            )

            def _to_capstor_path(p):
                if not isinstance(p, str) or not p:
                    return p
                if p.startswith('/'):
                    return p
                if p.startswith('capstor/'):
                    return os.path.join(CAPSTOR_ROOT, p[len('capstor/'):])
                if p.startswith('./capstor/'):
                    return os.path.join(CAPSTOR_ROOT, p[len('./capstor/'):])
                return p

            for key in ['experiment_dir', 'checkpoint_dir', 'log_dir', 'save_dir', 'pred_save_dir']:
                if hasattr(self, key):
                    setattr(self, key, _to_capstor_path(getattr(self, key)))

            # Create relevant directories (main process only)
            os.makedirs(self.experiment_dir, exist_ok=True)
            os.makedirs(self.checkpoint_dir, exist_ok=True)
            os.makedirs(self.log_dir, exist_ok=True)
            os.makedirs(self.save_dir, exist_ok=True)
            os.makedirs(self.pred_save_dir, exist_ok=True)

            logger.info(f"Configuration successfully loaded from {yaml_path}")
        except Exception as e:
            logger.error(f"Error loading config from {yaml_path}: {e}")
            raise e
        
    def to_dict(self):
        """Convert configuration to dictionary."""
        return {k: v for k, v in self.__dict__.items() 
                if not k.startswith('_') and not callable(v)}
    
    def save(self, path):
        """Save configuration to YAML file."""
        
        try:
            with open(path, 'w') as f:
                yaml.dump(self.to_dict(), f, default_flow_style=False)
            logger.info(f"Configuration saved to {path}")
        except Exception as e:
            logger.error(f"Error saving config to {path}: {e}")
            raise e








def parse_args_and_load_config():
    """Parse command line arguments and load configuration."""
    parser = argparse.ArgumentParser(description='Efficiency metrics evaluation.')
    
    # Config file argument
    parser.add_argument('--config', type=str, required=False,
                       help='Path to config YAML file')
    
    # Model arguments
    parser.add_argument('--model_type', type=str, help='Ablation model type',
                      choices=['no_rl', 'mean_pool', 'attention_pool', 'single_task',
                              '3d_cnn', 'cnn_lstm', 'video_transformer'])
    
    # Data arguments
    parser.add_argument('--video_folder', type=str, help='Path to video folder')
    
    # Model loading arguments
    parser.add_argument('--model_weights', type=str, help='Path to model weights')
    parser.add_argument('--best_model_path', type=str, help='Path to best model for evaluation')
    parser.add_argument('--resume_from_checkpoint', type=str, help='Path to checkpoint to resume from')
    
    args = parser.parse_args()
    
    # Create config with defaults
    config = Config()
    
    # Load YAML config if provided
    if args.config:
        if os.path.exists(args.config):
            config.load_from_yaml(args.config)
        else:
            if is_main_process():
                logger.error(f"Config file not found: {args.config}")
            raise FileNotFoundError(f"Config file not found: {args.config}")
    else:
        if is_main_process():
            logger.info("No config file provided, using defaults")
    
    # Override with command-line arguments
    if args.model_type is not None:
        config.model_type = args.model_type
    
    if args.lr is not None:
        config.learning_rate = args.lr
    
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    
    if args.epochs is not None:
        config.num_epochs = args.epochs
    
    if args.seed is not None:
        config.seed = args.seed
    
    if args.video_folder is not None:
        config.video_folder = args.video_folder
    
    if args.model_weights is not None:
        config.model_weights = args.model_weights
    
    if args.best_model_path is not None:
        config.best_model_path = args.best_model_path
    
    if args.resume_from_checkpoint is not None:
        config.resume_from_checkpoint = args.resume_from_checkpoint
    
    if args.eval_only:
        config.train = False
    
    return config

if __name__ == "__main__":
    config = parse_args_and_load_config()
    logger.info("Final configuration:")
    logger.info(config)