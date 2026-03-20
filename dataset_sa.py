import os
import glob
import torch
import numpy as np
import pandas as pd
from PIL import Image
import cv2
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.transforms.functional as F
from typing import Dict, List, Tuple, Union, Optional, Callable
from collections import defaultdict, OrderedDict
from functools import lru_cache
import random
from PIL import ImageEnhance, ImageFilter

NUM_PATH_CLASSES = 4

SA_SITE_ALIASES = {
    # TrUST raw site codes -> internal canonical site codes.
    "RA": "APXD",
    "RAS": "QASD",
    "RAI": "QAID",
    "RLS": "QSLD",
    "RL": "QLD",
    "LA": "APXG",
    "LAS": "QASG",
    "LAI": "QAIG",
    "LLS": "QSLG",
    "LL": "QLG",
    "RPS": "QPSD",
    "RPI": "QPID",
    "LPS": "QPSG",
    "LPI": "QPIG",

    # Internal aliases used elsewhere in the repo.
    "QLID": "QLD",
    "QLIG": "QLG",

    # A few rare variants present in the processed SA metadata.
    "LLI": "QLG",
    "RP1": "QPID",
}

SA_CANONICAL_SITES = {
    "QAID",
    "QAIG",
    "QASD",
    "QASG",
    "QLD",
    "QLG",
    "QPID",
    "QPIG",
    "QPSD",
    "QPSG",
    "APXD",
    "APXG",
    "QSLD",
    "QSLG",
    "SAD",
    "SLD",
    "SAG",
    "SLG",
    "SPD",
    "SPG",
    "UNKNOWN",
}


def normalize_sa_site_code(site: object) -> str:
    code = str(site or "UNKNOWN").strip().upper() or "UNKNOWN"
    mapped = SA_SITE_ALIASES.get(code, code)
    if mapped == "<PAD>":
        return "UNKNOWN"
    if mapped in SA_CANONICAL_SITES:
        return mapped
    return "UNKNOWN"

class UltrasoundPreprocessing(object):
    def __call__(self, img):
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(1.2)
        return img

class UltrasoundNoiseAugment(object):
    def __call__(self, tensor):
        speckle = torch.randn_like(tensor) * 0.2
        tensor = tensor * (1 + speckle)
        tensor = torch.clamp(tensor, 0, 1)
        return tensor

class TemporallyConsistentTransforms:
    def __init__(self, 
                 resize_size=(224, 224),
                 degrees=25,
                 translate=(0.15, 0.15),
                 scale=(0.65, 1.45),
                 brightness=0.3,
                 contrast=0.3,
                 blur_kernel_size=3,
                 blur_sigma=(0.1, 0.5),
                 noise_std=0.2,
                 augment_prob=0.5,
                 blur_prob=0.2,
                 mean=[0.45, 0.45, 0.45],
                 std=[0.225, 0.225, 0.225]):
        
        self.resize_size = resize_size
        self.degrees = degrees
        self.translate = translate
        self.scale = scale
        self.brightness = brightness
        self.contrast = contrast
        self.blur_kernel_size = blur_kernel_size
        self.blur_sigma = blur_sigma
        self.noise_std = noise_std
        self.augment_prob = augment_prob
        self.blur_prob = blur_prob
        self.mean = torch.tensor(mean).view(1, 3, 1, 1)
        self.std = torch.tensor(std).view(1, 3, 1, 1)
    
    def __call__(self, video_frames):
        if not video_frames:
            return torch.empty(0, 3, *self.resize_size)
        
        augment_params = self._sample_augmentation_parameters()
        
        transformed_frames = []
        for frame in video_frames:
            frame = self._apply_resize(frame)
            frame = self._apply_contrast_enhancement(frame, augment_params)
            frame = self._apply_affine_transform(frame, augment_params)
            frame = self._apply_color_jitter(frame, augment_params)
            frame = self._apply_gaussian_blur(frame, augment_params)
            
            frame_tensor = F.to_tensor(frame)
            transformed_frames.append(frame_tensor)
        
        video_tensor = torch.stack(transformed_frames)
        video_tensor = self._apply_noise(video_tensor, augment_params)
        video_tensor = self._apply_normalization(video_tensor)
        
        return video_tensor
    
    def _sample_augmentation_parameters(self):
        params = {}
        
        if random.random() < self.augment_prob:
            params['apply_affine'] = True
            params['angle'] = random.uniform(-self.degrees, self.degrees)
            params['translate'] = (
                random.uniform(-self.translate[0], self.translate[0]),
                random.uniform(-self.translate[1], self.translate[1])
            )
            params['scale'] = random.uniform(self.scale[0], self.scale[1])
        else:
            params['apply_affine'] = False
        
        params['brightness_factor'] = random.uniform(
            max(0, 1 - self.brightness), 1 + self.brightness
        )
        params['contrast_factor'] = random.uniform(
            max(0, 1 - self.contrast), 1 + self.contrast
        )
        
        if random.random() < self.blur_prob:
            params['apply_blur'] = True
            params['blur_sigma'] = random.uniform(self.blur_sigma[0], self.blur_sigma[1])
        else:
            params['apply_blur'] = False
        
        params['noise_multiplier'] = torch.randn(1, 1, 1, 1) * self.noise_std
        
        return params
    
    def _apply_resize(self, frame):
        return F.resize(frame, self.resize_size)
    
    def _apply_contrast_enhancement(self, frame, params):
        enhancer = ImageEnhance.Contrast(frame)
        return enhancer.enhance(1.2)
    
    def _apply_affine_transform(self, frame, params):
        if not params['apply_affine']:
            return frame
        
        width, height = frame.size
        translate_pixels = (
            int(params['translate'][0] * width),
            int(params['translate'][1] * height)
        )
        
        return F.affine(
            frame,
            angle=params['angle'],
            translate=translate_pixels,
            scale=params['scale'],
            shear=0,
            fill=0
        )
    
    def _apply_color_jitter(self, frame, params):
        frame = F.adjust_brightness(frame, params['brightness_factor'])
        frame = F.adjust_contrast(frame, params['contrast_factor'])
        return frame
    
    def _apply_gaussian_blur(self, frame, params):
        if not params['apply_blur']:
            return frame
        
        return frame.filter(ImageFilter.GaussianBlur(radius=params['blur_sigma']))
    
    def _apply_noise(self, video_tensor, params):
        speckle = params['noise_multiplier'] * torch.randn_like(video_tensor)
        video_tensor = video_tensor * (1 + speckle)
        return torch.clamp(video_tensor, 0, 1)
    
    def _apply_normalization(self, video_tensor):
        return (video_tensor - self.mean) / self.std

class SimpleVideoTransforms:
    def __init__(self, 
                 resize_size=(224, 224),
                 mean=[0.485, 0.456, 0.406],
                 std=[0.229, 0.224, 0.225]):
        
        self.resize_size = resize_size
        self.mean = torch.tensor(mean).view(1, 3, 1, 1)
        self.std = torch.tensor(std).view(1, 3, 1, 1)
    
    def __call__(self, video_frames):
        if not video_frames:
            return torch.empty(0, 3, *self.resize_size)
        
        transformed_frames = []
        for frame in video_frames:
            frame = F.resize(frame, self.resize_size)
            frame_tensor = F.to_tensor(frame)
            transformed_frames.append(frame_tensor)
        
        video_tensor = torch.stack(transformed_frames)
        video_tensor = (video_tensor - self.mean) / self.std
        
        return video_tensor

class PatientLevelDataset(Dataset):
    
    def __init__(self, 
                 root_dir: str,
                 labels_csv: str,
                 file_metadata_csv: str,
                 image_folder: str = 'images',
                 video_folder: str = 'videos',
                 split: str = 'train',
                 split_csv: Optional[str] = None,
                 image_transforms: Optional[Callable] = None,
                 depth_filter: str = 'all',
                 video_transforms: Optional[Callable] = None,
                 mode: str = 'video',
                 frame_sampling: int = 16,
                 selected_sites: Optional[List[str]] = None,
                 cache_size: int = 100,
                 files_per_site: Optional[Union[int, str]] = 'all', 
                 site_order: Optional[List[str]] = None,   
                 pad_missing_sites: bool = True,           
                 max_sites: Optional[int] = None           
                 ):
        
        self.root_dir = root_dir
        self.image_folder = image_folder
        self.video_folder = video_folder
        self.split = split
        self.depth_filter = depth_filter
        self.mode = mode.lower()

        
        self.files_per_site = files_per_site
        self.site_order = site_order
        self.pad_missing_sites = pad_missing_sites
        self.max_sites = max_sites
        self._uses_controlled_selection = (
            files_per_site != 'all' and pad_missing_sites
        )

        if self.mode not in ['video', 'image', 'both']:
            raise ValueError(f"Invalid mode: {mode}. Use 'video', 'image', or 'both'.")
        self.frame_sampling = frame_sampling
        self.selected_sites = selected_sites
        
        self.image_transforms = image_transforms or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.video_transforms = video_transforms or SimpleVideoTransforms()
        
        self.site_mapping = OrderedDict([
            ("<PAD>", 0),

            # Right hemithorax
            ("QAID", 1),   # Right anterior-inferior
            ("QAIG", 2),   # Left anterior-inferior
            ("QASD", 3),   # Right anterior-superior
            ("QASG", 4),   # Left anterior-superior
            ("QLD", 5),    # Right lateral-lower
            ("QLG", 6),    # Left lateral-lower
            ("QPID", 7),   # Right posterior-inferior
            ("QPIG", 8),   # Left posterior-inferior
            ("QPSD", 9),   # Right posterior-superior
            ("QPSG", 10),  # Left posterior-superior
            ("APXD", 11),  # Right apical
            ("APXG", 12),  # Left apical
            ("QSLD", 13),  # Right lateral-superior
            ("QSLG", 14),  # Left lateral-superior

            # Remaining entries
            ("SAD", 15),
            ("SLD", 16),
            ("SAG", 17),
            ("SLG", 18),
            ("SPD", 19),
            ("SPG", 20),
            ("UNKNOWN", 21),
        ])

        if self.selected_sites:
            normalized_selected_sites = []
            for site in self.selected_sites:
                normalized_site = normalize_sa_site_code(site)
                if normalized_site not in normalized_selected_sites:
                    normalized_selected_sites.append(normalized_site)
            self.selected_sites = normalized_selected_sites

        if self.site_order is not None:
            normalized_site_order = []
            for site in self.site_order:
                normalized_site = normalize_sa_site_code(site)
                if normalized_site not in normalized_site_order:
                    normalized_site_order.append(normalized_site)
            self.site_order = normalized_site_order

        self.labels_df = pd.read_csv(labels_csv)
        self.file_metadata_df = pd.read_csv(file_metadata_csv)

        self.optional_label_names = ['Pneumonia', 'Covid']

        required_labels = ['TB Label']
        missing_required = [label for label in required_labels if label not in self.labels_df.columns]
        if missing_required:
            available_columns = list(self.labels_df.columns)
            missing_str = ", ".join(missing_required)
            raise ValueError(
                f"Required label(s) {missing_str} not found in labels CSV. Available columns: {available_columns}"
            )

        for optional_label in self.optional_label_names:
            if optional_label not in self.labels_df.columns:
                self.labels_df[optional_label] = -1

        self.labels_df['patient_id'] = self.labels_df['record_id'].astype(str).str.strip()
        self.file_metadata_df['patient_id'] = self.file_metadata_df['Patient ID'].astype(str).str.strip()

        self.labels_df['patient_id_padded'] = self.labels_df['record_id'].astype(str).str.strip()
        self.file_metadata_df['patient_id_padded'] = self.file_metadata_df['Patient ID'].astype(str).str.strip()
        
        # Only pad if the ID is a pure number (not "25-4" format)
        def safe_zfill(pid):
            pid_str = str(pid).strip()
            if pid_str.isdigit():
                return pid_str.zfill(3)
            return pid_str
        
        self.labels_df['patient_id_padded'] = self.labels_df['patient_id_padded'].apply(safe_zfill)
        self.file_metadata_df['patient_id_padded'] = self.file_metadata_df['patient_id_padded'].apply(safe_zfill)

        # Extract numeric IDs for matching (handles "25-XXX" vs "27-XXX" mismatch)
        def extract_numeric_id(pid):
            pid_str = str(pid).strip()
            if '-' in pid_str:
                return pid_str.split('-')[-1]
            return pid_str
        
        self.labels_df['numeric_id'] = self.labels_df['patient_id'].apply(extract_numeric_id)
        self.file_metadata_df['numeric_id'] = self.file_metadata_df['patient_id'].apply(extract_numeric_id)

        label_patients = set(self.labels_df['patient_id'].unique())
        label_patients_padded = set(self.labels_df['patient_id_padded'].unique())
        label_patients_numeric = set(self.labels_df['numeric_id'].unique())
        file_patients = set(self.file_metadata_df['patient_id'].unique())
        file_patients_padded = set(self.file_metadata_df['patient_id_padded'].unique())
        file_patients_numeric = set(self.file_metadata_df['numeric_id'].unique())
        
        matches_unpadded = len(label_patients.intersection(file_patients))
        matches_padded = len(label_patients_padded.intersection(file_patients_padded))
        matches_numeric = len(label_patients_numeric.intersection(file_patients_numeric))
        
        # Choose best matching strategy
        # Priority: direct match > numeric match > padded match
        # Apply this regardless of split_csv to ensure labels and file_metadata can be matched
        if matches_unpadded > 0:
            # Direct match works, use it
            pass
        elif matches_numeric > matches_padded and matches_numeric > 0:
            # Numeric matching works best (handles "25-XXX" vs "27-XXX")
            print(f"Using numeric ID matching: {matches_numeric} matches (unpadded: {matches_unpadded}, padded: {matches_padded})")
            self.labels_df['patient_id'] = self.labels_df['numeric_id']
            self.file_metadata_df['patient_id'] = self.file_metadata_df['numeric_id']
            # Debug
            print(f"After conversion - Labels sample IDs: {sorted(list(self.labels_df['patient_id'].unique()))[:10]}")
            print(f"After conversion - File metadata sample IDs: {sorted(list(self.file_metadata_df['patient_id'].unique()))[:10]}")
        elif matches_padded > matches_unpadded:
            # Padded matching works better
            self.labels_df['patient_id'] = self.labels_df['patient_id_padded']
            self.file_metadata_df['patient_id'] = self.file_metadata_df['patient_id_padded']

        if split_csv and split != 'all':
            self._load_split_info(split_csv)
            self._filter_by_split()

        self._apply_depth_filter()
        self.file_metadata_df['normalized_site'] = self.file_metadata_df['Site'].apply(normalize_sa_site_code)

        self._extract_site_labels()
        
        if self.selected_sites:
            before_count = len(self.file_metadata_df)
            self.file_metadata_df = self.file_metadata_df[
                self.file_metadata_df['normalized_site'].isin(self.selected_sites)
            ]
            after_count = len(self.file_metadata_df)
            if after_count == 0 and before_count > 0:
                print(f"⚠️  WARNING: After site filter, 0 files remain (was {before_count})")
        
        self._index_files()
        
        # Debug: Check file_metadata before grouping
        print(f"Before _group_by_patient: {len(self.file_metadata_df)} files, {len(self.file_metadata_df['patient_id'].unique())} unique patients")
        if len(self.file_metadata_df) > 0:
            print(f"  Sample patient IDs: {sorted(list(self.file_metadata_df['patient_id'].unique()))[:10]}")
        
        self.patients = self._group_by_patient()
        
        print(f"After _group_by_patient: {len(self.patients)} patients returned")
        if len(self.patients) > 0:
            sample_patient_ids = [p['patient_id'] for p in self.patients[:10]]
            print(f"  Sample patient IDs: {sample_patient_ids}")
        
        if len(self.patients) == 0:
            print(f"⚠️  WARNING: Dataset has 0 patients after all filtering")
            print(f"   File metadata rows: {len(self.file_metadata_df)}")
            if len(self.file_metadata_df) > 0:
                print(f"   Sample patient IDs in file_metadata: {list(self.file_metadata_df['patient_id'].unique())[:5]}")
                print(f"   Sample splits: {list(self.file_metadata_df['split'].unique()) if 'split' in self.file_metadata_df.columns else 'N/A'}")
        
        self.video_cache = lru_cache(maxsize=cache_size)(self._load_video_uncached)
        self.image_cache = lru_cache(maxsize=cache_size)(self._load_image)
    
    def _load_split_info(self, split_csv):
        split_df = pd.read_csv(split_csv, dtype=str).fillna("")
        
        if 'train_ids' in split_df.columns and 'test_ids' in split_df.columns:
            # Filter out empty strings and NaN values properly
            train_ids = split_df['train_ids'].astype(str).replace('', np.nan).dropna().tolist()
            test_ids = split_df['test_ids'].astype(str).replace('', np.nan).dropna().tolist()
            val_ids = []
            if 'valid_ids' in split_df.columns:
                val_ids = split_df['valid_ids'].astype(str).replace('', np.nan).dropna().tolist()
            elif 'val_ids' in split_df.columns:
                val_ids = split_df['val_ids'].astype(str).replace('', np.nan).dropna().tolist()
            
            # Normalize IDs - remove any whitespace and convert to string
            train_ids = [str(pid).strip() for pid in train_ids if pid and str(pid).strip() and str(pid).strip() != 'nan']
            test_ids = [str(pid).strip() for pid in test_ids if pid and str(pid).strip() and str(pid).strip() != 'nan']
            val_ids = [str(pid).strip() for pid in val_ids if pid and str(pid).strip() and str(pid).strip() != 'nan']
            
            # If file_metadata is using numeric IDs (after matching conversion), convert split IDs too
            using_numeric_ids = False
            if hasattr(self, 'file_metadata_df') and 'numeric_id' in self.file_metadata_df.columns:
                # Check if patient_id column now contains numeric IDs
                sample_pid = str(self.file_metadata_df['patient_id'].iloc[0])
                if '-' not in sample_pid and sample_pid.isdigit():
                    using_numeric_ids = True
                    print(f"Converting split IDs to numeric format to match file_metadata")
            
            def to_numeric_if_needed(pid):
                if using_numeric_ids:
                    if '-' in pid:
                        return pid.split('-')[-1]
                    return pid
                return pid
            
            self.patient_splits = {}
            for pid in train_ids:
                key = to_numeric_if_needed(pid)
                self.patient_splits[key] = 'train'
            for pid in test_ids:
                key = to_numeric_if_needed(pid)
                self.patient_splits[key] = 'test'
            for pid in val_ids:
                key = to_numeric_if_needed(pid)
                self.patient_splits[key] = 'val'
        else:
            self.patient_splits = dict(zip(split_df['patient_id'].astype(str), split_df['split']))
        
        # Normalize patient IDs in file_metadata_df for matching
        self.file_metadata_df['patient_id'] = self.file_metadata_df['patient_id'].astype(str).str.strip()
        
        # Also try matching with padded versions
        if 'patient_id_padded' in self.file_metadata_df.columns:
            self.file_metadata_df['patient_id_padded'] = self.file_metadata_df['patient_id_padded'].astype(str).str.strip()
        
        # Ensure numeric_id column exists for matching
        if 'numeric_id' not in self.file_metadata_df.columns:
            def extract_numeric_id(pid):
                pid_str = str(pid).strip()
                if '-' in pid_str:
                    return pid_str.split('-')[-1]
                return pid_str
            self.file_metadata_df['numeric_id'] = self.file_metadata_df['patient_id'].apply(extract_numeric_id)
        
        # Map split - try multiple ID formats for matching
        def get_split(pid, pid_numeric=None):
            pid_str = str(pid).strip()
            
            # Try direct match first
            if pid_str in self.patient_splits:
                return self.patient_splits[pid_str]
            
            # Try numeric match if available (handles "25-XXX" vs "27-XXX")
            if pid_numeric and str(pid_numeric).strip() in self.patient_splits:
                return self.patient_splits[str(pid_numeric).strip()]
            
            # Try matching with different formats - STRICT matching only
            for split_id in self.patient_splits.keys():
                split_id_str = str(split_id).strip()
                # Direct match (already checked above, but keep for safety)
                if pid_str == split_id_str:
                    return self.patient_splits[split_id]
                
                # Try matching just the number part (e.g., "27-243" matches "27-243" or "243")
                if '-' in split_id_str:
                    num_part = split_id_str.split('-')[-1]
                    # STRICT: Only match if our pid IS the number part (no endswith)
                    if pid_str == num_part:
                        return self.patient_splits[split_id]
                    # STRICT: Match if both have prefixes and number parts are identical
                    if '-' in pid_str and pid_str.split('-')[-1] == num_part:
                        return self.patient_splits[split_id]
                
                # Try matching with padding - STRICT equality only
                try:
                    if pid_str.isdigit():
                        pid_padded = pid_str.zfill(3)
                        # STRICT: Only exact match, no endswith
                        if pid_padded == split_id_str:
                            return self.patient_splits[split_id]
                        # Also try if split_id is just the number
                        if '-' in split_id_str:
                            split_num = split_id_str.split('-')[-1]
                            if pid_padded == split_num:
                                return self.patient_splits[split_id]
                except:
                    pass
            
            return 'unknown'
        
        # Apply split mapping with numeric ID support
        self.file_metadata_df['split'] = self.file_metadata_df.apply(
            lambda row: get_split(row['patient_id'], row.get('numeric_id')), 
            axis=1
        )
        
        # Debug: Check matching results
        unknown_count = (self.file_metadata_df['split'] == 'unknown').sum()
        if unknown_count > 0 and len(self.file_metadata_df) > 0:
            print(f"⚠️  INFO: {unknown_count}/{len(self.file_metadata_df)} file metadata rows have 'unknown' split after initial matching")
            sample_unknown = self.file_metadata_df[self.file_metadata_df['split'] == 'unknown']['patient_id'].head(5).tolist()
            print(f"   Sample unknown patient IDs: {sample_unknown}")
            print(f"   Sample patient_splits keys: {list(self.patient_splits.keys())[:5]}")
        
        # Also try matching with padded IDs if available
        if 'patient_id_padded' in self.file_metadata_df.columns:
            def get_split_padded(pid):
                pid_str = str(pid).strip()
                if pid_str in self.patient_splits:
                    return self.patient_splits[pid_str]
                return None
            
            # Update unknown splits with padded ID matches
            unknown_mask = self.file_metadata_df['split'] == 'unknown'
            if unknown_mask.any():
                padded_splits = self.file_metadata_df.loc[unknown_mask, 'patient_id_padded'].map(get_split_padded)
                self.file_metadata_df.loc[unknown_mask & padded_splits.notna(), 'split'] = padded_splits[unknown_mask & padded_splits.notna()]
                
                # Debug: Check if padded matching helped
                still_unknown = (self.file_metadata_df['split'] == 'unknown').sum()
                if still_unknown < unknown_count:
                    print(f"   ✓ Padded matching helped: {unknown_count - still_unknown} additional matches found")
        
    
    def _filter_by_split(self):
        split_value = self.split
        if split_value == 'val' and not any(self.file_metadata_df['split'] == 'val'):
            split_value = 'valid'
        elif split_value == 'valid' and not any(self.file_metadata_df['split'] == 'valid'):
            split_value = 'val'
        
        before_count = len(self.file_metadata_df)
        before_patients = len(self.file_metadata_df['patient_id'].unique())
        self.file_metadata_df = self.file_metadata_df[
            self.file_metadata_df['split'] == split_value
        ]
        after_count = len(self.file_metadata_df)
        after_patients = len(self.file_metadata_df['patient_id'].unique())
        
        print(f"After split filter '{split_value}': {after_count} files, {after_patients} patients (was {before_count} files, {before_patients} patients)")
        if after_patients > 0:
            print(f"  Sample patient IDs after filter: {sorted(list(self.file_metadata_df['patient_id'].unique()))[:10]}")
        
        if after_count == 0:
            print(f"⚠️  WARNING: After filtering by split '{split_value}', 0 files remain (was {before_count})")
            print(f"   Available splits in data: {self.file_metadata_df['split'].unique() if 'split' in self.file_metadata_df.columns else 'N/A'}")
            if hasattr(self, 'patient_splits'):
                print(f"   Patient splits loaded: {len(self.patient_splits)} patients")
                sample_splits = {k: v for k, v in list(self.patient_splits.items())[:5]}
                print(f"   Sample patient splits: {sample_splits}")

    def _apply_depth_filter(self):
        before_count = len(self.file_metadata_df)
        before_patients = len(self.file_metadata_df['patient_id'].unique())
        if self.depth_filter == 'all':
            pass
        elif self.depth_filter == '5':
            self.file_metadata_df = self.file_metadata_df[
                self.file_metadata_df['Depth'].astype(int) < 10
            ]
        elif self.depth_filter == '15':
            self.file_metadata_df = self.file_metadata_df[
                self.file_metadata_df['Depth'].astype(int) > 10
            ]
        else:
            raise ValueError(f"Invalid depth_filter: {self.depth_filter}. Use 'all', '5', or '15'.")
        after_count = len(self.file_metadata_df)
        after_patients = len(self.file_metadata_df['patient_id'].unique())
        
        print(f"After depth filter '{self.depth_filter}': {after_count} files, {after_patients} patients (was {before_count} files, {before_patients} patients)")
        if after_patients > 0 and after_patients != before_patients:
            print(f"  Sample patient IDs after depth filter: {sorted(list(self.file_metadata_df['patient_id'].unique()))[:10]}")
        
        if after_count == 0 and before_count > 0:
            print(f"⚠️  WARNING: After depth filter '{self.depth_filter}', 0 files remain (was {before_count})")
    
    def _extract_site_labels(self):
        
        self.site_codes = set()
        label_exclusion = ['record_id', 'patient_id', 'patient_id_padded', 'numeric_id', 'data_source', 'TB Label'] + self.optional_label_names

        for col in self.labels_df.columns:
            if '_' in col and col not in label_exclusion:
                site = normalize_sa_site_code(col.split('_')[0])
                self.site_codes.add(site)
        
        self.finding_columns = [
            col for col in self.labels_df.columns
            if not col.endswith('_severity')
            and col not in label_exclusion
            and '_' in col
        ]
        
        self.patient_labels = {}
        for _, row in self.labels_df.iterrows():
            patient_id = row['patient_id']
            label_entry = {'TB Label': row.get('TB Label', -1)}
            for optional_label in self.optional_label_names:
                label_entry[f'{optional_label} Label'] = row.get(optional_label, -1)
            self.patient_labels[patient_id] = label_entry
        
        # Debug: Check patient_labels creation
        if len(self.patient_labels) > 0:
            sample_label_ids = list(self.patient_labels.keys())[:5]
            print(f"Created patient_labels: {len(self.patient_labels)} patients, sample IDs: {sample_label_ids}")
            # Check if specific test patients are in patient_labels
            test_check = ['100', '101', '105', '109']
            for tid in test_check:
                if tid in self.patient_labels:
                    print(f"  ✓ '{tid}' is in patient_labels")
                else:
                    print(f"  ✗ '{tid}' is NOT in patient_labels")

        all_label_columns = ['TB Label'] + self.optional_label_names

        for label_name in all_label_columns:
            lookup_key = 'TB Label' if label_name == 'TB Label' else f'{label_name} Label'
            self.file_metadata_df[label_name] = self.file_metadata_df['patient_id'].map(
                lambda pid: self.patient_labels.get(pid, {}).get(lookup_key, -1)
            )
        
        self.site_labels = {}
        
        for _, row in self.labels_df.iterrows():
            patient_id = row['patient_id']
            
            if patient_id not in self.site_labels:
                self.site_labels[patient_id] = {}
            
            for site in self.site_codes:
                if site not in self.site_labels[patient_id]:
                    self.site_labels[patient_id][site] = {'findings': {}}
                
                self.site_labels[patient_id][site]['findings'] = {}
                
                site_prefixes = [site]
                if site == 'QLD':
                    site_prefixes.append('QLID')
                elif site == 'QLG':
                    site_prefixes.append('QLIG')
                
                for prefix in site_prefixes:
                    for col in self.finding_columns:
                        if col.startswith(f"{prefix}_") and not pd.isna(row[col]) and row[col] != -1:
                            finding_type = col[len(prefix)+1:]

                            if finding_type not in {
                                'A-line',
                                'B-lines',
                                'Confluent B-lines',
                                'small Consolidations or Nodules',
                                'large Consolidations',
                                'Pleural effusion',
                            }:
                                continue

                            mapped_value = int(row[col])
                            existing_value = self.site_labels[patient_id][site]['findings'].get(finding_type, -1)
                            self.site_labels[patient_id][site]['findings'][finding_type] = max(
                                existing_value,
                                mapped_value,
                            )
        
        # Apply site findings lookup
        self.file_metadata_df['site_findings'] = self.file_metadata_df.apply(
            lambda row: self._get_site_findings(
                row['patient_id'],
                row.get('normalized_site', row['Site']),
            ),
            axis=1,
        )
        
        # Debug: Check how many files have pathology findings
        files_with_findings = self.file_metadata_df['site_findings'].apply(lambda x: len(x) > 0).sum()
        total_files = len(self.file_metadata_df)
        print(f"Pathology labels loaded: {files_with_findings}/{total_files} files have site-level pathology findings")
        
        # Show sample of unique site names from file_metadata
        unique_sites = self.file_metadata_df['Site'].unique()[:15]
        print(f"  Unique site names in file_metadata (sample): {list(unique_sites)}")
        normalized_sites = self.file_metadata_df['normalized_site'].unique()[:15]
        print(f"  Normalized site names in file_metadata (sample): {list(normalized_sites)}")
        
        # Show sample of site codes extracted from labels
        sample_site_codes = list(self.site_codes)[:10]
        print(f"  Site codes extracted from pathology labels: {sample_site_codes}")
    
    def _get_site_findings(self, patient_id, site):
        mapped_site = normalize_sa_site_code(site)
        if patient_id in self.site_labels and mapped_site in self.site_labels[patient_id]:
            return self.site_labels[patient_id][mapped_site].get('findings', {})
        return {}
    
    def _index_files(self):
        self.image_paths = {}
        self.video_paths = {}
        
        if self.mode in ['image', 'both']:
            images_dir = os.path.join(self.root_dir, self.image_folder)
            if os.path.exists(images_dir):
                image_files = glob.glob(os.path.join(images_dir, "*.png"))
                for img_file in image_files:
                    filename = os.path.basename(img_file)
                    file_key = os.path.splitext(filename)[0]
                    self.image_paths[file_key] = img_file
        
        if self.mode in ['video', 'both']:
            # Check if video_folder is absolute path
            if os.path.isabs(self.video_folder):
                videos_dir = self.video_folder
            else:
                videos_dir = os.path.join(self.root_dir, self.video_folder)
            
            if os.path.exists(videos_dir):
                video_files = glob.glob(os.path.join(videos_dir, "*.mp4"))
                for vid_file in video_files:
                    filename = os.path.basename(vid_file)
                    file_key = os.path.splitext(filename)[0]
                    self.video_paths[file_key] = vid_file
            else:
                # Try to find videos in common locations
                possible_locations = [
                    os.path.join(self.root_dir, "videos"),
                    os.path.join(self.root_dir, "Videos"),
                    "/capstor/scratch/cscs/mbarbiere/ultr-ai/LusBeninVideos",
                    "/capstor/scratch/cscs/tbrokowski/ultr-ai/LusBeninVideos",
                ]
                for loc in possible_locations:
                    if os.path.exists(loc):
                        video_files = glob.glob(os.path.join(loc, "*.mp4"))
                        for vid_file in video_files:
                            filename = os.path.basename(vid_file)
                            file_key = os.path.splitext(filename)[0]
                            self.video_paths[file_key] = vid_file
                        break
        
        self.file_metadata_df['file_key'] = self.file_metadata_df.apply(
            lambda row: f"{row['Patient ID']}_{row['Site']}_{row['Depth']}_{row['Count']}", 
            axis=1
        )
    
    def _group_by_patient(self):
        patients = {}
        
        if self.site_order is not None:
            ordered_sites = self.site_order
        else:
            ordered_sites = [site for site in self.site_mapping.keys() if site != "<PAD>"]
        
        if self.max_sites is not None:
            ordered_sites = ordered_sites[:self.max_sites]
        
        patient_groups = self.file_metadata_df.groupby('patient_id')
        
        skipped_count = 0
        for patient_id, group in patient_groups:
            patient_labels = self.patient_labels.get(patient_id, {})
            
            # Try to find patient with different ID format if not found
            if not patient_labels:
                found_labels = None
                for label_pid in self.patient_labels.keys():
                    if str(patient_id).strip() == str(label_pid).strip():
                        found_labels = self.patient_labels[label_pid]
                        break
                    # Try matching number part if format is "25-4"
                    if '-' in str(patient_id) and '-' in str(label_pid):
                        if str(patient_id).split('-')[-1] == str(label_pid).split('-')[-1]:
                            found_labels = self.patient_labels[label_pid]
                            break
                
                if found_labels:
                    patient_labels = found_labels
                else:
                    skipped_count += 1
                    if skipped_count <= 5:  # Only print first 5
                        print(f"  Skipping patient '{patient_id}' - no labels found")
                    continue
            
            # Check if all labels are -1 (no valid labels for any task)
            # Only skip if ALL task labels are -1 AND we're in training mode
            if patient_labels and all(label == -1 for label in patient_labels.values()):
                if self.split == 'train':
                    continue
            
            # Patient has valid labels or we're allowing unlabeled data
            
            if patient_id not in patients:
                patients[patient_id] = {
                    'patient_labels': patient_labels,
                    'files': []
                }
            
            patient_files_by_site = {}
            for _, file_info in group.iterrows():
                file_key = file_info['file_key']
                raw_site = str(file_info['Site']).upper()
                site = file_info.get('normalized_site', normalize_sa_site_code(raw_site))
                
                file_exists = False
                if (self.mode == 'video' and file_key in self.video_paths) or \
                   (self.mode == 'image' and file_key in self.image_paths) or \
                   (self.mode == 'both' and (file_key in self.video_paths or file_key in self.image_paths)):
                    file_exists = True
                
                if file_exists:
                    if site not in patient_files_by_site:
                        patient_files_by_site[site] = []
                    
                    depth = int(file_info['Depth'])
                    site_findings = file_info.get('site_findings', {})
                    
                    patient_files_by_site[site].append({
                        'file_key': file_key,
                        'site': site,
                        'raw_site': raw_site,
                        'site_index': self.site_mapping.get(site, self.site_mapping['UNKNOWN']),
                        'depth': depth,
                        'site_findings': site_findings,
                        'has_video': file_key in self.video_paths,
                        'has_image': file_key in self.image_paths,
                        'is_real': True
                    })
            
            selected_files = []
            
            for site in ordered_sites:
                site_files = list(patient_files_by_site.get(site, []))
                
                if site_files:
                    if self.files_per_site == 'all':
                        selected_files.extend(site_files)
                    else:
                        site_files.sort(key=lambda x: x['depth'])
                        actual_files = site_files[:self.files_per_site]
                        selected_files.extend(actual_files)
                        
                        num_padding_needed = self.files_per_site - len(actual_files)
                        for pad_idx in range(num_padding_needed):
                            selected_files.append({
                                'file_key': f"PAD_{site}_{len(actual_files) + pad_idx}",
                                'site': site,
                                'site_index': self.site_mapping.get(site, 0),
                                'depth': -1,
                                'site_findings': {},
                                'has_video': False,
                                'has_image': False,
                                'is_real': False
                            })
                
                elif self.pad_missing_sites and self.files_per_site != 'all':
                    for pad_idx in range(self.files_per_site):
                        selected_files.append({
                            'file_key': f"PAD_{site}_{pad_idx}",
                            'site': site,
                            'site_index': self.site_mapping.get(site, 0),
                            'depth': -1,
                            'site_findings': {},
                            'has_video': False,
                            'has_image': False,
                            'is_real': False
                        })
                        
            patients[patient_id]['files'] = selected_files
        
        patients = {pid: data for pid, data in patients.items() if data['files']}
        
        patient_samples = [
            {'patient_id': pid, **data}
            for pid, data in patients.items()
        ]
        
        return patient_samples
    
    def _load_image(self, image_path):
        try:
            image = Image.open(image_path).convert('RGB')
            if self.image_transforms:
                image = self.image_transforms(image)
            return image
        except Exception as e:
            return None
    
    def _load_video_uncached(self, video_path):
        try:
            cap = cv2.VideoCapture(video_path)
            
            if not cap.isOpened():
                return torch.empty(0, 3, 224, 224)
            
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            if frame_count <= 0:
                return torch.empty(0, 3, 224, 224)
            
            indices = np.linspace(0, frame_count - 1, self.frame_sampling, dtype=int)
            frames = []
            
            for i in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                ret, frame = cap.read()
                if ret:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame = Image.fromarray(frame)
                    frames.append(frame)
            
            cap.release()
            
            if not frames:
                return torch.empty(0, 3, 224, 224)
            
            if self.video_transforms:
                video_tensor = self.video_transforms(frames)
            else:
                simple_transforms = SimpleVideoTransforms()
                video_tensor = simple_transforms(frames)
            
            return video_tensor
            
        except Exception as e:
            return torch.empty(0, 3, 224, 224)
    
    def _get_findings_onehot(self, findings_dict, num_classes=NUM_PATH_CLASSES):
        one_hot = torch.full((num_classes,), -1.0)
        
        if not findings_dict or not isinstance(findings_dict, dict):
            return one_hot
        
        if findings_dict.get('A-line') in [0, 1]:
            one_hot[0] = float(findings_dict['A-line'])
        
        if findings_dict.get('large Consolidations') in [0, 1]:
            one_hot[1] = float(findings_dict['large Consolidations'])

        if findings_dict.get('Pleural effusion') in [0, 1]:
            one_hot[2] = float(findings_dict['Pleural effusion'])
        
        other_values = [
            findings_dict[key]
            for key in ['B-lines', 'Confluent B-lines', 'small Consolidations or Nodules']
            if findings_dict.get(key) in [0, 1]
        ]
        
        if any(value == 1 for value in other_values):
            one_hot[3] = 1
        elif other_values:
            one_hot[3] = 0
        
        return one_hot

    def create_site_order_presets(self):
        presets = {}
        
        presets.update({
            'anatomical': ['QASD', 'QAID', 'QASG', 'QAIG', 'QLD', 'QLG', 'QPSD', 'QPID', 'QPSG', 'QPIG', 'APXD', 'APXG', 'QSLD', 'QSLG'],
            'anterior_first': ['QASD', 'QASG', 'QAID', 'QAIG', 'QLD', 'QLG', 'QPSD', 'QPSG', 'QPID', 'QPIG'],
            'bilateral_pairs': ['QASD', 'QASG', 'QAID', 'QAIG', 'QLD', 'QLG', 'QPSD', 'QPSG', 'QPID', 'QPIG'],
            'standard_first': ['QASD', 'QAID', 'QLD', 'QASG', 'QAIG', 'QLG', 'QPSD', 'QPID', 'QPSG', 'QPIG'],
            'sweep_first': ['SAD', 'SLD', 'SAG', 'SLG', 'SPD', 'SPG', 'QASD', 'QAID', 'QLD', 'QASG', 'QAIG', 'QLG', 'QPSD', 'QPID', 'QPSG', 'QPIG'],
            'anterior_posterior': ['QASD', 'QAID', 'SAD', 'QASG', 'QAIG', 'SAG', 'QLD', 'SLD', 'QLG', 'SLG', 'QPSD', 'QPID', 'SPD', 'QPSG', 'QPIG', 'SPG'],
            'sweep_only': ['SAD', 'SLD', 'SAG', 'SLG', 'SPD', 'SPG'],
            'standard_only': ['QASD', 'QAID', 'QLD', 'QASG', 'QAIG', 'QLG', 'QPSD', 'QPID', 'QPSG', 'QPIG']
        })
        
        return presets

    def get_site_statistics(self):
        stats = {
            'total_patients': len(self.patients),
            'site_coverage': {},
            'files_per_patient': [],
            'real_files_per_patient': []
        }
        
        for site in self.site_mapping.keys():
            if site != "<PAD>":
                stats['site_coverage'][site] = {
                    'patients_with_site': 0,
                    'total_files': 0,
                    'real_files': 0,
                    'padded_files': 0
                }
        
        for patient in self.patients:
            files = patient['files']
            stats['files_per_patient'].append(len(files))
            
            real_files = sum(1 for f in files if f['is_real'])
            stats['real_files_per_patient'].append(real_files)
            
            patient_sites = set()
            for file_info in files:
                mapped_site = normalize_sa_site_code(file_info['site'])
                    
                if mapped_site in stats['site_coverage']:
                    stats['site_coverage'][mapped_site]['total_files'] += 1
                    if file_info['is_real']:
                        stats['site_coverage'][mapped_site]['real_files'] += 1
                        patient_sites.add(mapped_site)
                    else:
                        stats['site_coverage'][mapped_site]['padded_files'] += 1
            
            for site in patient_sites:
                stats['site_coverage'][site]['patients_with_site'] += 1
        
        return stats

    def print_dataset_summary(self):
        pass
    
    def __len__(self):
        return len(self.patients)
    
    def __getitem__(self, idx):
        patient = self.patients[idx]
        patient_id = patient['patient_id']
        patient_labels = patient['patient_labels']
        files = patient['files']
        
        site_data = []
        
        for file_info in files:
            file_key = file_info['file_key']
            site = file_info['site']
            site_index = file_info['site_index']
            depth = file_info['depth']
            is_real = file_info['is_real']
            site_findings = file_info['site_findings']
            
            findings_onehot = self._get_findings_onehot(site_findings)
            
            video = None
            image = None
            
            if is_real:
                has_video = file_info['has_video']
                has_image = file_info['has_image']
                
                if has_video and self.mode in ['video', 'both']:
                    video_path = self.video_paths[file_key]
                    video = self.video_cache(video_path)
                    
                    if video is None or video.numel() == 0:
                        is_real = False
                
                if has_image and self.mode in ['image', 'both']:
                    image_path = self.image_paths[file_key]
                    image = self.image_cache(image_path)
                    
                    if image is None:
                        is_real = False
            
            if not is_real:
                if self.mode in ['video', 'both']:
                    video = torch.zeros(self.frame_sampling, 3, 224, 224)
                if self.mode in ['image', 'both']:
                    image = torch.zeros(3, 224, 224)
                findings_onehot = torch.full((NUM_PATH_CLASSES,), -1.0)
            
            site_data.append({
                'file_key': file_key,
                'site': site,
                'site_index': site_index,
                'depth': depth,
                'findings_onehot': findings_onehot,
                'site_findings': site_findings,
                'video': video,
                'image': image,
                'is_real': is_real
            })
        
        if not site_data:
            return {
                'patient_id': patient_id,
                'tb_label': patient_labels.get('TB Label', -1),
                'pneumonia_label': patient_labels.get('Pneumonia Label', -1),
                'covid_label': patient_labels.get('Covid Label', -1),
                'num_sites': 0,
                'site_indices': torch.zeros(1, dtype=torch.long),
                'site_videos': torch.zeros(1, self.frame_sampling, 3, 224, 224),
                'site_images': torch.zeros(1, 3, 224, 224),
                'site_findings': torch.full((1, NUM_PATH_CLASSES), -1.0),
                'is_real_mask': torch.zeros(1, dtype=torch.bool),
                'is_valid': False,
                '_uses_controlled_selection': self._uses_controlled_selection
            }
        
        num_sites = len(site_data)
        site_indices = torch.tensor([s['site_index'] for s in site_data], dtype=torch.long)
        site_findings = torch.stack([s['findings_onehot'] for s in site_data])
        is_real_mask = torch.tensor([s['is_real'] for s in site_data], dtype=torch.bool)
        
        if self.mode in ['video', 'both']:
            videos = [s['video'] for s in site_data if s['video'] is not None]
            if videos and all(v.shape[0] == videos[0].shape[0] for v in videos):
                site_videos = torch.stack(videos)
            else:
                max_frames = max(v.shape[0] for v in videos) if videos else self.frame_sampling
                C, H, W = (3, 224, 224)
                site_videos = torch.zeros(num_sites, max_frames, C, H, W)
                for i, video in enumerate(videos):
                    if video is not None:
                        site_videos[i, :video.shape[0]] = video
        else:
            site_videos = torch.zeros(num_sites, self.frame_sampling, 3, 224, 224)
        
        if self.mode in ['image', 'both']:
            images = [s['image'] for s in site_data if s['image'] is not None]
            if images:
                site_images = torch.stack(images)
            else:
                site_images = torch.zeros(num_sites, 3, 224, 224)
        else:
            site_images = torch.zeros(num_sites, 3, 224, 224)
        
        result = {
            'patient_id': patient_id,
            'tb_label': patient_labels.get('TB Label', -1),
            'pneumonia_label': patient_labels.get('Pneumonia Label', -1),
            'covid_label': patient_labels.get('Covid Label', -1),
            'num_sites': num_sites,
            'site_indices': site_indices,
            'site_videos': site_videos,
            'site_images': site_images,
            'site_findings': site_findings,
            'is_real_mask': is_real_mask,
            'is_valid': True,
            '_uses_controlled_selection': self._uses_controlled_selection
        }
        
        return result


def collate_patient_batch(batch):
    
    valid_batch = [sample for sample in batch if sample['is_valid']]
    
    if not valid_batch:
        return {
            'patient_ids': [],
            'tb_labels': torch.zeros(0, dtype=torch.long),
            'pneumonia_labels': torch.zeros(0, dtype=torch.long),
            'covid_labels': torch.zeros(0, dtype=torch.long),
            'site_indices': torch.zeros(0, dtype=torch.long),
            'site_counts': torch.zeros(0, dtype=torch.long),
            'site_videos': torch.zeros(0, 0, 0, 3, 224, 224),
            'site_images': torch.zeros(0, 0, 3, 224, 224),
            'site_findings': torch.zeros(0, 0, NUM_PATH_CLASSES),
            'site_masks': torch.zeros(0, 0, dtype=torch.bool),
            '_mask_type': 'empty'
        }
    
    uses_controlled_selection = valid_batch[0].get('_uses_controlled_selection', False)
    
    batch_size = len(valid_batch)
    patient_ids = [sample['patient_id'] for sample in valid_batch]
    tb_labels = torch.tensor([sample['tb_label'] for sample in valid_batch], dtype=torch.long)
    pneumonia_labels = torch.tensor([sample['pneumonia_label'] for sample in valid_batch], dtype=torch.long)
    covid_labels = torch.tensor([sample['covid_label'] for sample in valid_batch], dtype=torch.long)
    
    max_sites = max(sample['num_sites'] for sample in valid_batch)
    
    site_counts = torch.tensor([sample['num_sites'] for sample in valid_batch], dtype=torch.long)
    site_indices = torch.zeros(batch_size, max_sites, dtype=torch.long)
    site_findings = torch.full((batch_size, max_sites, NUM_PATH_CLASSES), -1.0)
    
    batch_padding_masks = torch.zeros(batch_size, max_sites, dtype=torch.bool)
    real_data_masks = torch.zeros(batch_size, max_sites, dtype=torch.bool)
    
    video_shape = valid_batch[0]['site_videos'].shape[1:]
    image_shape = valid_batch[0]['site_images'].shape[1:]
    
    site_videos = torch.zeros(batch_size, max_sites, *video_shape)
    site_images = torch.zeros(batch_size, max_sites, *image_shape)
    
    for i, sample in enumerate(valid_batch):
        num_sites = sample['num_sites']
        site_indices[i, :num_sites] = sample['site_indices']
        site_findings[i, :num_sites] = sample['site_findings']
        site_videos[i, :num_sites] = sample['site_videos']
        site_images[i, :num_sites] = sample['site_images']
        
        batch_padding_masks[i, :num_sites] = True
        
        real_data_masks[i, :num_sites] = sample['is_real_mask']
    
    if uses_controlled_selection:
        adaptive_site_masks = real_data_masks
        mask_type = 'real_data'
    else:
        adaptive_site_masks = batch_padding_masks  
        mask_type = 'batch_padding'
    
    result = {
        'patient_ids': patient_ids,
        'tb_labels': tb_labels,
        'pneumonia_labels': pneumonia_labels,
        'covid_labels': covid_labels,
        'site_indices': site_indices,
        'site_counts': site_counts,
        'site_videos': site_videos,
        'site_images': site_images,
        'site_findings': site_findings,
        'site_masks': adaptive_site_masks,
        'batch_padding_masks': batch_padding_masks,
        'real_data_masks': real_data_masks,
        '_mask_type': mask_type
    }
    
    return result


class LungUltrasoundDataModule:
    
    def __init__(self, 
                 root_dir: str,
                 labels_csv: str,
                 file_metadata_csv: str,
                 image_folder: str = 'images',
                 video_folder: str = 'videos',
                 split_csv: Optional[str] = None,
                 batch_size: int = 8,
                 num_workers: int = 4,
                 frame_sampling: int = 16,
                 depth_filter: str = 'all',
                 cache_size: int = 100,
                 files_per_site: Optional[Union[int, str]] = 'all',  
                 site_order: Optional[List[str]] = None,   
                 pad_missing_sites: bool = True,           
                 max_sites: Optional[int] = None):
        
        self.root_dir = root_dir
        self.labels_csv = labels_csv
        self.file_metadata_csv = file_metadata_csv
        self.image_folder = image_folder
        self.video_folder = video_folder
        self.split_csv = split_csv
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.frame_sampling = frame_sampling
        self.cache_size = cache_size
        self.depth_filter = depth_filter
        self.files_per_site = files_per_site
        self.site_order = site_order
        self.pad_missing_sites = pad_missing_sites
        self.max_sites = max_sites
        
        self.image_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.video_transforms = SimpleVideoTransforms()

        self.train_video_transforms = TemporallyConsistentTransforms(
            degrees=25,
            translate=(0.15, 0.15),
            scale=(0.65, 1.45),
            brightness=0.3,
            contrast=0.3,
            blur_sigma=(0.1, 0.5),
            noise_std=0.2,
            augment_prob=0.5,
            blur_prob=0.2,
            mean=[0.45, 0.45, 0.45],
            std=[0.225, 0.225, 0.225]
        )
    
    def setup(self, stage=None):
        if stage == 'patient_level' or stage is None:
            self.patient_train = PatientLevelDataset(
                root_dir=self.root_dir,
                labels_csv=self.labels_csv,
                file_metadata_csv=self.file_metadata_csv,
                image_folder=self.image_folder,
                video_folder=self.video_folder,
                split='train',
                split_csv=self.split_csv,
                image_transforms=self.image_transforms,
                video_transforms=self.train_video_transforms,
                mode='video',
                frame_sampling=self.frame_sampling,
                depth_filter=self.depth_filter,
                cache_size=self.cache_size,
                files_per_site=self.files_per_site,
                site_order=self.site_order,
                pad_missing_sites=self.pad_missing_sites,
                max_sites=self.max_sites
            )
            
            self.patient_val = PatientLevelDataset(
                root_dir=self.root_dir,
                labels_csv=self.labels_csv,
                file_metadata_csv=self.file_metadata_csv,
                image_folder=self.image_folder,
                video_folder=self.video_folder,
                split='val',
                split_csv=self.split_csv,
                image_transforms=self.image_transforms,
                video_transforms=self.video_transforms,
                mode='video',
                frame_sampling=self.frame_sampling,
                depth_filter=self.depth_filter,
                cache_size=self.cache_size,
                files_per_site=self.files_per_site,
                site_order=self.site_order,
                pad_missing_sites=self.pad_missing_sites,
                max_sites=self.max_sites
            )
            
            self.patient_test = PatientLevelDataset(
                root_dir=self.root_dir,
                labels_csv=self.labels_csv,
                file_metadata_csv=self.file_metadata_csv,
                image_folder=self.image_folder,
                video_folder=self.video_folder,
                split='test',
                split_csv=self.split_csv,
                image_transforms=self.image_transforms,
                video_transforms=self.video_transforms,
                mode='video',
                depth_filter=self.depth_filter,
                frame_sampling=self.frame_sampling,
                cache_size=self.cache_size,
                files_per_site=self.files_per_site,
                site_order=self.site_order,
                pad_missing_sites=self.pad_missing_sites,
                max_sites=self.max_sites
            )
    
    def patient_level_dataloader(self, split='train'):
        if split == 'train':
            dataset = self.patient_train
            shuffle = True
        elif split == 'val':
            dataset = self.patient_val
            shuffle = False
        elif split == 'test':
            dataset = self.patient_test
            shuffle = False
        else:
            raise ValueError(f"Invalid split: {split}")
        
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=collate_patient_batch, 
            prefetch_factor=2,
        )
    
    def train_dataloader(self, stage='patient_level'):
        if stage == 'patient_level':
            return self.patient_level_dataloader('train')
        else:
            raise ValueError(f"Invalid stage: {stage}")
    
    def val_dataloader(self, stage='patient_level'):
        if stage == 'patient_level':
            return self.patient_level_dataloader('val')
        else:
            raise ValueError(f"Invalid stage: {stage}")
    
    def test_dataloader(self):
        return self.patient_level_dataloader('test')


def analyze_batch_masks(batch):
    site_masks = batch['site_masks']
    if 'real_data_masks' in batch:
        real_data_masks = batch['real_data_masks']

def analyze_batch_labels(batch):
    pass
