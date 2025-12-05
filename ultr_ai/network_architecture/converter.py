"""Convert the current pytorch model to tf.js"""
import argparse
import torch

from ultr_ai.config import load_config
from ultr_ai.network_architecture import create_ablation_model

def convert_to_tfjs(model, output_dir):
    print(f"Converting model to TF.js format at {output_dir}...")



def main():
    parser = argparse.ArgumentParser(description='Convert model formats.')
    
    # Data arguments
    # parser.add_argument('--video_folder', type=str, help='Name of the video folder within the data directory')
    
    # Single fold processing
    parser.add_argument('--model-type', type=str, default='original')
    parser.add_argument('--config', type=str, help='Path to config file')
    parser.add_argument('--model', type=str, help='Path to model checkpoint')
    parser.add_argument('--output-dir', type=str, default='ULTR-CLIP/results',
                       help='Output directory')
    parser.add_argument('--fold', type=int, default=0, help='Fold number')

    
    # Multi-fold processing
    # parser.add_argument('--process-all-folds', action='store_true',
    #                    help='Process all folds at once')
    # parser.add_argument('--config-pattern', type=str,
    #                    help='Pattern for config files (e.g., "configs/fold_{}.yaml")')
    # parser.add_argument('--model-pattern', type=str,
    #                    help='Pattern for model files (e.g., "checkpoints/fold_{}/best.pth")')
    # parser.add_argument('--num-folds', type=int, default=5,
    #                    help='Number of folds to process')
    
    args = parser.parse_args()

    # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Load config
    config = load_config(config_file=args.config)
    # Optional override for video folder
    # if args.video_folder:
    #     try:
    #         old_vf = getattr(config, 'video_folder', None)
    #     except Exception:
    #         old_vf = None
    #     setattr(config, 'video_folder', args.video_folder)
    #     print(f"Overriding video_folder: {old_vf} -> {args.video_folder}")

    # Load model
    model = create_ablation_model(args.model_type, config)
    print(f"Model initialized: {args.model_type}")
    checkpoint = torch.load(args.model, weights_only=False)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    print("Successfully loaded model")
    model.eval()

    # Convert to TF.js
    convert_to_tfjs(model, args.output_dir)
    
    # Setup data
    # data_module = LungUltrasoundDataModule(
    #     root_dir=config.root_dir,
    #     labels_csv=config.labels_csv,
    #     file_metadata_csv=config.file_metadata_csv,
    #     image_folder=config.image_folder,
    #     video_folder=config.video_folder,
    #     split_csv=config.split_csv,
    #     batch_size=1,
    #     num_workers=config.num_workers,
    #     frame_sampling=config.frame_sampling,
    #     depth_filter=config.depth_filter,
    #     cache_size=100,
    # )
    
    # data_module.setup(stage='patient_level')
    # videos = next(iter(data_module.patient_level_dataloader('test')))
    # print(videos["site_videos"].shape) # torch.Size([1, 24, 32, 3, 224, 224])

    # Compute efficiency metrics
    # num_params = compute_num_params(model)
    # dtype_counts = dtype_of_weights(model)

    # print(f"Number of parameters: {num_params}")
    # print(f"dtype counts: {dtype_counts}")
    # print(f"Estimated model size (MB): {compute_memory_usage(num_params, dtype_counts)}")
    # inference_time = compute_inference_time(model, data_module.patient_level_dataloader('test'), device)
    # print(f"Average inference time per batch (s): {inference_time}")



if __name__ == '__main__':
    main()