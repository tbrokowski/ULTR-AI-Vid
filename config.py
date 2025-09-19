import os
import yaml
import torch
from typing import List, Dict, Union, Optional
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


@dataclass
class MultiTaskConfig:
    """
    Comprehensive configuration class for multi-task lung ultrasound classification
    with configurable frame selection strategies.
    """
    
    # ==========================================
    # CORE CONFIGURATION
    # ==========================================
    
    # Experiment settings
    experiment_name: str = "multitask_experiment"
    experiment_dir: str = "./experiments/multitask"
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Training control
    train: bool = True
    evaluate_best_valid_model: bool = True
    
    # ==========================================
    # MULTI-TASK CONFIGURATION
    # ==========================================
    
    # Task selection - specify which tasks to train/evaluate
    active_tasks: List[str] = field(default_factory=lambda: ['TB Label'])  # Options: 'TB Label', 'Pneumonia Label', 'Covid Label'
    task_weights: Dict[str, float] = field(default_factory=lambda: {'TB Label': 1.0, 'Pneumonia Label': 1.0, 'Covid Label': 1.0})
    task_pos_weights: Dict[str, float] = field(default_factory=lambda: {'TB Label': 2.0, 'Pneumonia Label': 2.0, 'Covid Label': 2.0})
    
    # Pathology configuration
    use_pathology_loss: bool = True
    pathology_weight: float = 0.5
    pathology_pos_weights: List[float] = field(default_factory=lambda: [2.0, 4.0, 3.0])
    num_pathologies: int = 3
    pathology_classes: List[str] = field(default_factory=lambda: ['A-line', 'Large consolidations', 'Other Pathology'])
    
    # ==========================================
    # FRAME SELECTION STRATEGY
    # ==========================================
    
    # Frame selection strategy - 'random', 'attention', or 'RL'
    selection_strategy: str = 'RL'
    
    # RL-specific parameters (only used when selection_strategy == 'RL')
    patient_weight: float = 1.0
    correct_factor: float = 1.5
    incorrect_factor: float = 3.0
    reward_scale: float = 3.0
    entropy_weight: float = 0.01
    rl_clamp_neg: float = -10.0
    rl_clamp_pos: float = 10.0
    rl_accumulation_steps: int = 8
    
    # Temperature control (for RL and attention strategies)
    temperature: float = 2.0
    temperature_min: float = 0.1
    temperature_max: float = 3.0
    temperature_decay: float = 0.999
    
    # Actor-Critic parameters (RL only)
    actor_lr: float = 3e-4
    actor_weight_decay: float = 1e-5
    critic_lr: float = 1e-3
    critic_weight_decay: float = 1e-5
    gamma: float = 0.99
    gae_lambda: float = 0.95
    use_frame_history: bool = True
    frame_selector_max_norm: float = 1.0
    
    # ==========================================
    # DATA CONFIGURATION
    # ==========================================
    
    # Dataset paths
    root_dir: str = "/path/to/data"
    labels_csv: str = "/path/to/labels.csv"
    file_metadata_csv: str = "/path/to/metadata.csv"
    split_csv: Optional[str] = None
    image_folder: str = "images"
    video_folder: str = "videos"
    
    # Data loading
    batch_size: int = 8
    num_workers: int = 4
    frame_sampling: int = 16
    depth_filter: str = "all"  # 'all', '5', '15'
    
    # Site selection and ordering
    files_per_site: Union[int, str] = 'all'
    site_order: Optional[List[str]] = None
    pad_missing_sites: bool = True
    max_sites: Optional[int] = None
    num_sites: int = 15
    
    # ==========================================
    # MODEL ARCHITECTURE
    # ==========================================
    
    # Model dimensions
    hidden_dim: int = 512
    dropout_rate: float = 0.3
    num_classes: int = 1  # Binary classification for each task
    
    # Cross-site attention configuration
    use_cross_site_attention: bool = True
    cross_site_num_heads: int = 8
    use_site_positional_encoding: bool = True
    
    # CLIP backbone
    backbone: str = "clip"
    freeze_backbone: bool = True
    pretrained: bool = True
    local_weights_dir: str = "/gpfs/gibbs/project/hartley/tjb76/artstuff_OPTIMIZEDWOOOO/NetworkArchitecture/CLIP_weights"
    
    # ==========================================
    # TRAINING CONFIGURATION
    # ==========================================
    
    # Training schedule
    num_epochs: int = 100
    accumulation_steps: int = 4
    use_amp: bool = True
    
    # Learning rates and schedulers
    backbone_lr: float = 1e-5
    backbone_weight_decay: float = 1e-4
    backbone_T_0: int = 10
    backbone_T_mult: int = 2
    backbone_eta_min: float = 1e-7
    
    pathology_lr: float = 3e-4
    pathology_weight_decay: float = 1e-4
    pathology_T_0: int = 10
    pathology_T_mult: int = 2
    pathology_eta_min: float = 1e-6
    
    patient_pipeline_lr: float = 1e-4
    patient_pipeline_weight_decay: float = 1e-4
    patient_pipeline_T_0: int = 10
    patient_pipeline_T_mult: int = 2
    patient_pipeline_eta_min: float = 1e-6
    
    # Loss weights (for multi-task balancing)
    pos_weight: float = 2.0  # Default positive weight for binary classification
    
    # ==========================================
    # EVALUATION CONFIGURATION
    # ==========================================
    
    # Evaluation metric and goal
    eval_metric: str = "auc"  # Primary metric for model selection
    eval_metric_goal: str = "max"  # 'max' or 'min'
    early_stopping_patience: int = 20
    
    # ==========================================
    # CHECKPOINT AND RESUME
    # ==========================================
    
    # Model weights and resuming
    model_weights: Optional[str] = None
    best_model_path: Optional[str] = None
    resume_from_checkpoint: Optional[str] = None
    reset_optimizers: bool = False
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        
        # ENSURE PROPER TYPE CONVERSION FOR NUMERIC FIELDS
        # This is crucial when loading from YAML files where values might be strings
        numeric_fields = [
            'seed', 'hidden_dim', 'num_pathologies', 'num_sites', 'batch_size', 
            'num_workers', 'frame_sampling', 'num_classes', 'cross_site_num_heads',
            'num_epochs', 'accumulation_steps', 'backbone_T_0', 'backbone_T_mult',
            'pathology_T_0', 'pathology_T_mult', 'patient_pipeline_T_0', 
            'patient_pipeline_T_mult', 'early_stopping_patience', 'rl_accumulation_steps'
        ]
        
        float_fields = [
            'pathology_weight', 'patient_weight', 'correct_factor', 'incorrect_factor',
            'reward_scale', 'entropy_weight', 'rl_clamp_neg', 'rl_clamp_pos',
            'temperature', 'temperature_min', 'temperature_max', 'temperature_decay',
            'actor_lr', 'actor_weight_decay', 'critic_lr', 'critic_weight_decay',
            'gamma', 'gae_lambda', 'frame_selector_max_norm', 'dropout_rate',
            'backbone_lr', 'backbone_weight_decay', 'backbone_eta_min',
            'pathology_lr', 'pathology_weight_decay', 'pathology_eta_min',
            'patient_pipeline_lr', 'patient_pipeline_weight_decay', 
            'patient_pipeline_eta_min', 'pos_weight'
        ]
        
        # Convert numeric fields
        for field_name in numeric_fields:
            if hasattr(self, field_name):
                value = getattr(self, field_name)
                if value is not None and not isinstance(value, int):
                    try:
                        setattr(self, field_name, int(value))
                    except (ValueError, TypeError):
                        logger.warning(f"Could not convert {field_name}={value} to int")
        
        # Convert float fields
        for field_name in float_fields:
            if hasattr(self, field_name):
                value = getattr(self, field_name)
                if value is not None and not isinstance(value, float):
                    try:
                        setattr(self, field_name, float(value))
                    except (ValueError, TypeError):
                        logger.warning(f"Could not convert {field_name}={value} to float")
        
        # Convert list fields
        if hasattr(self, 'pathology_pos_weights') and self.pathology_pos_weights:
            self.pathology_pos_weights = [float(x) for x in self.pathology_pos_weights]
        
        # Convert dict fields
        if hasattr(self, 'task_weights') and self.task_weights:
            self.task_weights = {k: float(v) for k, v in self.task_weights.items()}
        
        if hasattr(self, 'task_pos_weights') and self.task_pos_weights:
            self.task_pos_weights = {k: float(v) for k, v in self.task_pos_weights.items()}
        
        # Convert boolean fields (in case they come as strings from YAML)
        bool_fields = [
            'train', 'evaluate_best_valid_model', 'use_pathology_loss', 
            'use_frame_history', 'pad_missing_sites', 'use_cross_site_attention',
            'use_site_positional_encoding', 'freeze_backbone', 'pretrained',
            'use_amp', 'reset_optimizers'
        ]
        
        for field_name in bool_fields:
            if hasattr(self, field_name):
                value = getattr(self, field_name)
                if isinstance(value, str):
                    # Handle string boolean values
                    if value.lower() in ['true', '1', 'yes', 'on']:
                        setattr(self, field_name, True)
                    elif value.lower() in ['false', '0', 'no', 'off']:
                        setattr(self, field_name, False)
                    else:
                        logger.warning(f"Could not convert {field_name}={value} to bool")
        
        # Validate active tasks
        valid_tasks = ['TB Label', 'Pneumonia Label', 'Covid Label']
        for task in self.active_tasks:
            if task not in valid_tasks:
                raise ValueError(f"Invalid task: {task}. Valid tasks: {valid_tasks}")
        
        # Validate frame selection strategy
        valid_strategies = ['random', 'attention', 'RL']
        if self.selection_strategy not in valid_strategies:
            raise ValueError(f"Invalid selection strategy: {self.selection_strategy}. Valid strategies: {valid_strategies}")
        
        # Validate task weights and pos_weights
        for task in self.active_tasks:
            if task not in self.task_weights:
                self.task_weights[task] = 1.0
                logger.warning(f"No task weight specified for {task}, using default 1.0")
            
            if task not in self.task_pos_weights:
                self.task_pos_weights[task] = 2.0
                logger.warning(f"No positive weight specified for {task}, using default 2.0")
        
        # Ensure experiment directory exists
        os.makedirs(self.experiment_dir, exist_ok=True)
        
        # Set device
        if self.device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA not available, switching to CPU")
            self.device = "cpu"
        
        # Log configuration summary
        logger.info(f"Multi-task configuration initialized:")
        logger.info(f"  Active tasks: {self.active_tasks}")
        logger.info(f"  Frame selection strategy: {self.selection_strategy}")
        logger.info(f"  Use pathology loss: {self.use_pathology_loss}")
        logger.info(f"  Device: {self.device}")
    
    
    def to_dict(self) -> Dict:
        """Convert config to dictionary for serialization."""
        return {
            key: getattr(self, key) for key in self.__dataclass_fields__.keys()
        }
    
    def save(self, filepath: str):
        """Save configuration to YAML file."""
        config_dict = self.to_dict()
        
        # Convert any non-serializable types
        for key, value in config_dict.items():
            if isinstance(value, torch.device):
                config_dict[key] = str(value)
        
        with open(filepath, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)
        
        logger.info(f"Configuration saved to {filepath}")
    
    @classmethod
    def from_dict(cls, config_dict: Dict):
        """Create config from dictionary."""
        # Filter out keys that are not part of the dataclass
        valid_keys = set(cls.__dataclass_fields__.keys())
        filtered_dict = {k: v for k, v in config_dict.items() if k in valid_keys}
        
        return cls(**filtered_dict)
    
    @classmethod
    def load(cls, filepath: str):
        """Load configuration from YAML file."""
        with open(filepath, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        return cls.from_dict(config_dict)
    
    def update_from_args(self, args):
        """Update configuration from command line arguments."""
        for key, value in vars(args).items():
            if hasattr(self, key) and value is not None:
                setattr(self, key, value)
                logger.info(f"Updated {key} from command line: {value}")


# ==========================================
# PRESET CONFIGURATIONS
# ==========================================

def get_tb_only_config(**kwargs) -> MultiTaskConfig:
    """Get configuration for TB-only training."""
    config = MultiTaskConfig(
        experiment_name="tb_only",
        active_tasks=['TB Label'],
        task_weights={'TB Label': 1.0},
        task_pos_weights={'TB Label': 2.0},
        use_pathology_loss=True,
        selection_strategy='RL',
        **kwargs
    )
    return config


def get_pneumonia_only_config(**kwargs) -> MultiTaskConfig:
    """Get configuration for Pneumonia-only training."""
    config = MultiTaskConfig(
        experiment_name="pneumonia_only",
        active_tasks=['Pneumonia Label'],
        task_weights={'Pneumonia Label': 1.0},
        task_pos_weights={'Pneumonia Label': 2.0},
        use_pathology_loss=True,
        selection_strategy='RL',
        **kwargs
    )
    return config


def get_covid_only_config(**kwargs) -> MultiTaskConfig:
    """Get configuration for Covid-only training."""
    config = MultiTaskConfig(
        experiment_name="covid_only",
        active_tasks=['Covid Label'],
        task_weights={'Covid Label': 1.0},
        task_pos_weights={'Covid Label': 2.0},
        use_pathology_loss=True,
        selection_strategy='RL',
        **kwargs
    )
    return config


def get_all_tasks_config(**kwargs) -> MultiTaskConfig:
    """Get configuration for training on all tasks."""
    config = MultiTaskConfig(
        experiment_name="all_tasks",
        active_tasks=['TB Label', 'Pneumonia Label', 'Covid Label'],
        task_weights={
            'TB Label': 1.0,
            'Pneumonia Label': 1.0,
            'Covid Label': 1.0
        },
        task_pos_weights={
            'TB Label': 2.0,
            'Pneumonia Label': 2.0,
            'Covid Label': 2.0
        },
        use_pathology_loss=True,
        selection_strategy='RL',
        **kwargs
    )
    return config


def get_tb_pneumonia_config(**kwargs) -> MultiTaskConfig:
    """Get configuration for TB + Pneumonia training."""
    config = MultiTaskConfig(
        experiment_name="tb_pneumonia",
        active_tasks=['TB Label', 'Pneumonia Label'],
        task_weights={
            'TB Label': 1.0,
            'Pneumonia Label': 1.0
        },
        task_pos_weights={
            'TB Label': 2.0,
            'Pneumonia Label': 2.0
        },
        use_pathology_loss=True,
        selection_strategy='RL',
        **kwargs
    )
    return config


def get_random_baseline_config(**kwargs) -> MultiTaskConfig:
    """Get configuration for random frame selection baseline."""
    config = MultiTaskConfig(
        experiment_name="random_baseline",
        active_tasks=['TB Label'],
        selection_strategy='random',
        use_pathology_loss=False,  # Usually disabled for baselines
        **kwargs
    )
    return config


def get_attention_baseline_config(**kwargs) -> MultiTaskConfig:
    """Get configuration for attention-based frame selection."""
    config = MultiTaskConfig(
        experiment_name="attention_baseline",
        active_tasks=['TB Label'],
        selection_strategy='attention',
        use_pathology_loss=True,
        **kwargs
    )
    return config


# ==========================================
# CONFIGURATION FACTORY
# ==========================================

def load_config(config_name: Optional[str] = None, config_file: Optional[str] = None, **kwargs) -> MultiTaskConfig:
    """
    Load configuration with multiple options.
    
    Args:
        config_name: Name of preset configuration
        config_file: Path to YAML configuration file
        **kwargs: Additional parameters to override
    
    Returns:
        MultiTaskConfig instance
    """
    
    if config_file and os.path.exists(config_file):
        # Load from file
        logger.info(f"Loading configuration from {config_file}")
        config = MultiTaskConfig.load(config_file)
        
        # Apply any overrides
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
        
        return config
    
    elif config_name:
        # Load preset configuration
        preset_configs = {
            'tb_only': get_tb_only_config,
            'pneumonia_only': get_pneumonia_only_config,
            'covid_only': get_covid_only_config,
            'all_tasks': get_all_tasks_config,
            'tb_pneumonia': get_tb_pneumonia_config,
            'random_baseline': get_random_baseline_config,
            'attention_baseline': get_attention_baseline_config,
        }
        
        if config_name in preset_configs:
            logger.info(f"Loading preset configuration: {config_name}")
            return preset_configs[config_name](**kwargs)
        else:
            logger.warning(f"Unknown preset configuration: {config_name}")
            logger.info(f"Available presets: {list(preset_configs.keys())}")
    
    # Default configuration
    logger.info("Using default configuration")
    return MultiTaskConfig(**kwargs)


# ==========================================
# COMMAND LINE INTERFACE
# ==========================================

def create_argument_parser():
    """Create argument parser for command line interface."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Multi-Task Lung Ultrasound Classification')
    
    # Configuration loading - CHANGED: --config-file to --config
    parser.add_argument('--config', type=str, help='Path to YAML configuration file')
    parser.add_argument('--config-name', type=str, 
                       choices=['tb_only', 'pneumonia_only', 'covid_only', 'all_tasks', 
                               'tb_pneumonia', 'random_baseline', 'attention_baseline'],
                       help='Preset configuration name')
    
    # Core settings
    parser.add_argument('--experiment-name', type=str, help='Experiment name')
    parser.add_argument('--experiment-dir', type=str, help='Experiment directory')
    parser.add_argument('--device', type=str, choices=['cpu', 'cuda'], help='Device to use')
    parser.add_argument('--seed', type=int, help='Random seed')
    
    # Multi-task settings
    parser.add_argument('--active-tasks', nargs='+', 
                       choices=['TB Label', 'Pneumonia Label', 'Covid Label'],
                       help='Active tasks to train/evaluate')
    parser.add_argument('--use-pathology-loss', action='store_true', help='Enable pathology loss')
    parser.add_argument('--no-pathology-loss', dest='use_pathology_loss', action='store_false', help='Disable pathology loss')
    
    # Frame selection strategy
    parser.add_argument('--selection-strategy', type=str, 
                       choices=['random', 'attention', 'RL'],
                       help='Frame selection strategy')
    
    # Data settings
    parser.add_argument('--root-dir', type=str, help='Root data directory')
    parser.add_argument('--labels-csv', type=str, help='Path to labels CSV')
    parser.add_argument('--file-metadata-csv', type=str, help='Path to file metadata CSV')
    parser.add_argument('--split-csv', type=str, help='Path to split CSV')
    parser.add_argument('--batch-size', type=int, help='Batch size')
    parser.add_argument('--num-workers', type=int, help='Number of data loading workers')
    
    # Training settings
    parser.add_argument('--num-epochs', type=int, help='Number of training epochs')
    parser.add_argument('--learning-rate', type=float, help='Learning rate')
    parser.add_argument('--use-amp', action='store_true', help='Use automatic mixed precision')
    parser.add_argument('--no-amp', dest='use_amp', action='store_false', help='Disable automatic mixed precision')
    
    # Cross-site attention configuration
    parser.add_argument('--use-cross-site-attention', action='store_true', help='Enable cross-site attention')
    parser.add_argument('--no-cross-site-attention', dest='use_cross_site_attention', action='store_false', help='Disable cross-site attention')
    parser.add_argument('--cross-site-num-heads', type=int, help='Number of attention heads for cross-site attention')
    parser.add_argument('--use-site-positional-encoding', action='store_true', help='Use positional encoding for sites')
    parser.add_argument('--no-site-positional-encoding', dest='use_site_positional_encoding', action='store_false', help='Disable site positional encoding')
    
    # Resume and evaluation
    parser.add_argument('--resume-from-checkpoint', type=str, help='Path to checkpoint to resume from')
    parser.add_argument('--model-weights', type=str, help='Path to pretrained model weights')
    parser.add_argument('--evaluate-only', action='store_true', help='Only evaluate, do not train')
    
    parser.set_defaults(use_pathology_loss=True, use_amp=True, use_cross_site_attention=True, use_site_positional_encoding=True)
    
    return parser


def parse_args_and_load_config():
    """Parse command line arguments and load configuration."""
    parser = create_argument_parser()
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(
        config_name=args.config_name,
        config_file=args.config  # CHANGED: was args.config_file
    )
    
    # Update from command line arguments
    config.update_from_args(args)
    
    # Handle evaluate-only mode
    if args.evaluate_only:
        config.train = False
        config.evaluate_best_valid_model = True
    
    return config


def main():
    """Main function for command line usage."""
    return parse_args_and_load_config()


if __name__ == "__main__":
    config = main()
    print("Configuration loaded successfully!")
    print(f"Active tasks: {config.active_tasks}")
    print(f"Frame selection strategy: {config.selection_strategy}")
    print(f"Use pathology loss: {config.use_pathology_loss}")