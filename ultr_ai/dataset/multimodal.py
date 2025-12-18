import pandas as pd
import numpy as np
import h5py
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.impute import SimpleImputer
import warnings
from typing import Dict, List, Tuple, Optional, Union
import logging
import os

warnings.filterwarnings('ignore')

class MultimodalTBDataset(Dataset):
    """Updated dataset that maintains exact compatibility with models.py"""
    
    def __init__(self, clinical_data_path: str, h5_file_path: str, site_data_path: str,
                 patient_ids: Optional[List[str]] = None, transform=None, 
                 fit_scalers: bool = True, clinical_scaler=None, is_training: bool = True,
                 imputation_strategy: str = 'median', use_robust_scaling: bool = True,
                 handle_outliers: bool = True, min_feature_count: int = 10):
        
        self.clinical_data_path = clinical_data_path
        self.h5_file_path = h5_file_path
        self.site_data_path = site_data_path
        self.transform = transform
        self.is_training = is_training
        self.imputation_strategy = imputation_strategy
        self.use_robust_scaling = use_robust_scaling
        self.handle_outliers = handle_outliers
        self.min_feature_count = min_feature_count
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Load and process data
        self.clinical_data = self._load_clinical_data()
        self.site_data = self._load_site_data()
        self.patient_embeddings = self._load_patient_embeddings()
        
        # Validate and filter patient IDs
        if patient_ids is not None:
            self.patient_ids = self._validate_patient_ids([str(pid) for pid in patient_ids])
        else:
            self.patient_ids = self._find_common_patient_ids()
        
        # CLEAN: Only use medically meaningful features (exclude technical/administrative noise)
        self.clinical_features, self.feature_order, self.clinical_scaler = self._process_clean_medical_features(
            fit_scalers, clinical_scaler
        )
        
        # Create clean medical feature schema
        self.feature_schema = self._create_clean_medical_schema()
        
        # Set clinical_dim (should be ~55-60 features now)
        self.clinical_dim = len(self.feature_order)
        
        self.logger.info(f"Dataset initialized with {len(self.patient_ids)} patients")
        self.logger.info(f"Clinical features shape: {self.clinical_features.shape}")
        self.logger.info(f"Clinical dimension: {self.clinical_dim}")
        self.logger.info(f"Feature schema domains: {list(self.feature_schema.keys())}")
        
        # Validate clinical features
        self._validate_clinical_features()
        
        if self.patient_embeddings:
            sample_embedding = list(self.patient_embeddings.values())[0]
            self.logger.info(f"Patient embeddings dimension: {sample_embedding.shape}")
    
    def _get_clean_medical_features(self) -> Dict[str, List[str]]:
        """Define only clinically meaningful features, exclude technical noise"""
        
        # Based on your actual column names from the CSV
        clean_medical_features = {
            'demographics': [
                'sexe',  # sex
                'age'    # age
            ],
            
            'symptoms': [
                'chief_complaint',
                # Symptom onset
                'fever_sympt_onset', 'cough_sympt_onset', 'dyspnea_sympt_onset', 
                'chestpain_symp_onset', 'hemoptysis_sympt_onset',
                # Accompanying symptoms (binary indicators)
                'accompanying_symptoms___1', 'accompanying_symptoms___2', 'accompanying_symptoms___3',
                'accompanying_symptoms___4', 'accompanying_symptoms___5', 'accompanying_symptoms___6',
                'accompanying_symptoms___7', 'accompanying_symptoms___8', 'accompanying_symptoms___9',
                'accompanying_symptoms___10', 'accompanying_symptoms___11', 'accompanying_symptoms___12',
                'accompanying_symptoms___13',
                # Symptom duration
                'fever_acc_sympt_duration', 'cough_acc_sympt_duration', 'weight_acc_sympt_duration',
                'dyspnea_acc_sympt_duration', 'chest_acc_sympt_duration', 'hemopt_acc_sympt_duration',
                'sweats_acc_sympt_duration', 'asthen_acc_sympt_duration', 'anorex_acc_sympt_duration',
                # Chronic symptoms
                'chronic_cough'
            ],
            
            'vital_signs': [
                # Basic vitals
                'poids_kg', 'taille_cm', 'temperature', 'frequence_cardiaque',
                'tension_arterielle_systol', 'tension_arterielle_diastol',
                'frequence_respiratoire', 'saturation_en_oxygene_spo2',
                'glasgow_coma_scale', 'bmi',
                # SOFA scores
                'qsofa_freqresp', 'qsofa_gcs', 'qsofa_systolbp', 'quick_sofa_score'
            ],
            
            'medical_history': [
                'antibiotique_av',           # previous antibiotics
                'previous_tb_diagnosis',     # previous TB
                'smoker',                    # smoking status
                'lung_diseases',             # lung diseases
                'diabetes',                  # diabetes
                'hypertension',              # hypertension
                'cardiopathy',               # heart disease
                'hiv',                       # HIV status
               # 'covid_contact',             # COVID contact
               # 'covid_vacc',                # COVID vaccination
               # 'previous_covid_test',       # previous COVID test
                'traditional_med_sought'     # traditional medicine
            ],
            
            'physical_exam': [
                'general_state',                # general condition
             #   'pitting_peripheral_edema',     # peripheral edema
              #  'cerv_and_or_axillary_adp'      # lymphadenopathy
            ],
            
            'laboratory': [
              #  'covid_pcr',                # COVID PCR result
                'hiv_test_v2',              # HIV test result
            #    'malaria_rdt',              # malaria rapid test
                'cd4_counts_result'         # CD4 count (if available)
            ],
            
            # Optional: Include lung auscultation findings (structured medical data)
            # 'auscultation': [
            #     # Only include the primary auscultation findings, not reader variations
            #     'crepitant___1', 'crepitant___2', 'crepitant___3', 'crepitant___4',
            #     'crepitant___5', 'crepitant___6', 'crepitant___7', 'crepitant___8',
            #     'sibilant___1', 'sibilant___2', 'sibilant___3', 'sibilant___4',
            #     'sibilant___5', 'sibilant___6', 'sibilant___7', 'sibilant___8',
            #     'ronchus___1', 'ronchus___2', 'ronchus___3', 'ronchus___4',
            #     'ronchus___5', 'ronchus___6', 'ronchus___7', 'ronchus___8'
            # ]
        }
        
        return clean_medical_features
    
    def _process_clean_medical_features(self, fit_scalers: bool = True, 
                                      clinical_scaler=None) -> Tuple[pd.DataFrame, List[str], object]:
        """Process only clean medical features, exclude all technical noise"""
        
        # Get clean medical feature definitions
        clean_features_dict = self._get_clean_medical_features()
        
        # Flatten to get all feature names
        all_clean_features = []
        for domain, features in clean_features_dict.items():
            all_clean_features.extend(features)
        
        self.logger.info(f"Defined {len(all_clean_features)} clean medical features")
        
        # Check which features actually exist in the data
        available_columns = set(self.clinical_data.columns)
        existing_features = [f for f in all_clean_features if f in available_columns]
        missing_features = [f for f in all_clean_features if f not in available_columns]
        
        self.logger.info(f"Found {len(existing_features)}/{len(all_clean_features)} defined features in data")
        if missing_features:
            self.logger.info(f"Missing features: {missing_features[:10]}{'...' if len(missing_features) > 10 else ''}")
        
        # Extract clinical data for our patients only
        clinical_subset = self.clinical_data[
            self.clinical_data['record_id'].isin(self.patient_ids)
        ].copy()
        
        if len(clinical_subset) == 0:
            raise ValueError("No clinical data found for any patient IDs!")
        
        # Create feature dataframe with only existing clean features
        df = clinical_subset[['record_id'] + existing_features].copy()
        df = df.set_index('record_id')
        
        self.logger.info(f"Processing {len(existing_features)} clean medical features for {len(df)} patients")
        
        # Process features with robust pipeline
        processed_features = []
        final_feature_names = []
        
        for col in existing_features:
            try:
                series = df[col].copy()
                
                # Convert to numeric, coercing errors to NaN
                if series.dtype == 'object':
                    series = pd.to_numeric(series, errors='coerce')
                
                # Check if column has any valid values
                valid_count = series.notna().sum()
                if valid_count < self.min_feature_count:
                    self.logger.warning(f"Column {col} has only {valid_count} valid values, skipping")
                    continue
                
                # Conservative outlier handling
                if self.handle_outliers and valid_count > 5:
                    q05 = series.quantile(0.05)
                    q95 = series.quantile(0.95)
                    if not pd.isna(q05) and not pd.isna(q95) and q05 != q95:
                        series = series.clip(lower=q05, upper=q95)
                
                # Impute missing values
                if self.imputation_strategy == 'median':
                    fill_value = series.median()
                elif self.imputation_strategy == 'mean':
                    fill_value = series.mean()
                else:  # mode
                    mode_vals = series.mode()
                    fill_value = mode_vals.iloc[0] if len(mode_vals) > 0 else 0.0
                
                if pd.isna(fill_value):
                    fill_value = 0.0
                
                series = series.fillna(fill_value)
                
                # Final safety checks
                series = series.replace([np.inf, -np.inf], 0.0)
                
                # Add minimal noise for zero-variance features
                if series.std() < 1e-10:
                    noise = np.random.normal(0, 1e-10, len(series))
                    series = series + noise
                
                processed_features.append(series)
                final_feature_names.append(col)
                
            except Exception as e:
                self.logger.warning(f"Error processing column {col}: {e}, skipping")
                continue
        
        if len(processed_features) == 0:
            raise ValueError("No features could be processed successfully!")
        
        # Combine processed features
        processed_df = pd.concat(processed_features, axis=1)
        processed_df.columns = final_feature_names
        
        # Final NaN check
        nan_count = processed_df.isnull().sum().sum()
        if nan_count > 0:
            self.logger.warning(f"Found {nan_count} remaining NaN values, filling with 0")
            processed_df = processed_df.fillna(0.0)
        
        # Consistent scaling
        try:
            if fit_scalers:
                if self.use_robust_scaling:
                    scaler = RobustScaler()
                else:
                    scaler = StandardScaler()
                
                data_array = processed_df.values.astype(np.float64)
                data_array = np.nan_to_num(data_array, nan=0.0, posinf=1.0, neginf=-1.0)
                
                # Add minimal noise to zero-variance features
                for i in range(data_array.shape[1]):
                    col_std = np.std(data_array[:, i])
                    if col_std < 1e-8:
                        noise = np.random.normal(0, 1e-8, data_array.shape[0])
                        data_array[:, i] += noise
                
                scaled_values = scaler.fit_transform(data_array)
                
            else:
                if clinical_scaler is None:
                    raise ValueError("clinical_scaler must be provided if fit_scalers is False")
                
                data_array = processed_df.values.astype(np.float64)
                data_array = np.nan_to_num(data_array, nan=0.0, posinf=1.0, neginf=-1.0)
                
                # Validate scaler compatibility
                if hasattr(clinical_scaler, 'n_features_in_'):
                    if clinical_scaler.n_features_in_ != data_array.shape[1]:
                        self.logger.warning(f"Scaler expects {clinical_scaler.n_features_in_} features, "
                                          f"got {data_array.shape[1]}. Creating new scaler.")
                        scaler = RobustScaler() if self.use_robust_scaling else StandardScaler()
                        scaled_values = scaler.fit_transform(data_array)
                    else:
                        scaled_values = clinical_scaler.transform(data_array)
                        scaler = clinical_scaler
                else:
                    try:
                        scaled_values = clinical_scaler.transform(data_array)
                        scaler = clinical_scaler
                    except Exception:
                        self.logger.warning("Error using provided scaler. Creating new scaler.")
                        scaler = RobustScaler() if self.use_robust_scaling else StandardScaler()
                        scaled_values = scaler.fit_transform(data_array)
            
            # Validate scaled output
            if np.isnan(scaled_values).any():
                self.logger.warning("NaN values after scaling, replacing with 0")
                scaled_values = np.nan_to_num(scaled_values, nan=0.0, posinf=1.0, neginf=-1.0)
            
            scaled_df = pd.DataFrame(
                scaled_values, 
                index=processed_df.index, 
                columns=final_feature_names
            )
            
            # Report scaling statistics
            final_mean = scaled_values.mean()
            final_std = scaled_values.std()
            
            self.logger.info(f"Clean scaling successful - Mean: {final_mean:.6f}, Std: {final_std:.6f}")
            self.logger.info(f"Final clean feature count: {len(final_feature_names)}")
            
            # Warn if scaling seems problematic
            if abs(final_mean) > 5 or final_std > 10:
                self.logger.warning(f"Unusual scaling detected! Mean: {final_mean:.6f}, Std: {final_std:.6f}")
            
        except Exception as e:
            self.logger.error(f"Error in scaling: {e}")
            raise
        
        return scaled_df, final_feature_names, scaler
    
    def _create_clean_medical_schema(self) -> Dict[str, List[str]]:
        """Create feature schema using only the clean medical features"""
        
        # Get clean feature definitions
        clean_features_dict = self._get_clean_medical_features()
        
        # Filter to only include features that exist in our processed data
        available_features = set(self.feature_order)
        filtered_schema = {}
        
        for domain, features in clean_features_dict.items():
            domain_features = [f for f in features if f in available_features]
            if domain_features:
                filtered_schema[domain] = domain_features
        
        # Log the clean schema
        self.logger.info(f"Created clean medical schema with {len(filtered_schema)} domains:")
        total_features = 0
        for domain, features in filtered_schema.items():
            self.logger.info(f"  {domain}: {len(features)} features")
            total_features += len(features)
        
        self.logger.info(f"Total clean medical features: {total_features}")
        
        # Should have NO 'other' category now!
        if 'other' in filtered_schema:
            self.logger.warning(f"WARNING: Still have 'other' category with {len(filtered_schema['other'])} features!")
        
        return filtered_schema
    
    def _validate_clinical_features(self):
        """Validate that clinical features are properly processed"""
        if self.clinical_features.shape[0] == 0:
            raise ValueError("No clinical features available!")
        
        # Check for all-zero features
        zero_features = (self.clinical_features == 0).all(axis=0).sum()
        if zero_features == self.clinical_features.shape[1]:
            self.logger.warning("WARNING: All clinical features are zero!")
        elif zero_features > 0:
            self.logger.info(f"{zero_features}/{self.clinical_features.shape[1]} clinical features are all zero")
        
        # Check for NaN
        nan_count = np.isnan(self.clinical_features.values).sum()
        if nan_count > 0:
            self.logger.warning(f"WARNING: {nan_count} NaN values found in clinical features")
        
        # Check variance
        feature_stds = self.clinical_features.std()
        low_variance_features = (feature_stds < 1e-6).sum()
        if low_variance_features > 0:
            self.logger.info(f"{low_variance_features}/{len(feature_stds)} features have very low variance")
        
        self.logger.info(f"Clean medical features validation: mean={self.clinical_features.values.mean():.6f}, "
                        f"std={self.clinical_features.values.std():.6f}")
    
    def _load_clinical_data(self) -> pd.DataFrame:
        """Load and preprocess clinical data"""
        df = pd.read_csv(self.clinical_data_path)
        df['record_id'] = df['record_id'].astype(str)
        
        self.logger.info(f"Loaded clinical data: {df.shape}")
        self.logger.info(f"Missing data percentage: {(df.isnull().sum().sum() / df.size) * 100:.2f}%")
        
        return df
    
    def _load_site_data(self) -> pd.DataFrame:
        """Load and aggregate site data"""
        df = pd.read_csv(self.site_data_path)
        df['patient_id'] = df['patient_id'].astype(str)
        
        # Aggregate site data by patient
        aggregation_dict = {
            'tb_label': 'first',
            'tb_logit': 'mean', 
            'tb_prob': 'mean',
            'tb_pred': 'first',
            'a_lines_finding': 'mean',
            'b_lines_finding': 'mean',
            'small_consolidations_finding': 'mean',
            'large_consolidations_finding': 'mean',
            'pleural_effusion_finding': 'mean',
            'mil_attention': 'mean'
        }
        
        # Only include columns that exist
        existing_agg_dict = {k: v for k, v in aggregation_dict.items() if k in df.columns}
        
        # Count observations per patient
        site_counts = df.groupby('patient_id').size().reset_index(name='site_observation_count')
        
        # Aggregate data
        aggregated = df.groupby('patient_id').agg(existing_agg_dict).reset_index()
        aggregated = aggregated.merge(site_counts, on='patient_id', how='left')
        
        self.logger.info(f"Loaded site data: {len(df)} observations from {len(aggregated)} patients")
        
        return aggregated
    
    def _load_patient_embeddings(self) -> Dict[str, torch.Tensor]:
        """Load patient embeddings from H5 file"""
        embeddings = {}
        try:
            with h5py.File(self.h5_file_path, 'r') as f:
                if 'patient_features' in f:
                    for pid in f['patient_features'].keys():
                        pid_str = str(pid)
                        embeddings[pid_str] = torch.tensor(
                            f['patient_features'][pid][:], dtype=torch.float32
                        )
                else:
                    self.logger.warning(f"'patient_features' not found in H5 file")
                    
            self.logger.info(f"Loaded embeddings for {len(embeddings)} patients")
                
        except Exception as e:
            self.logger.error(f"Error loading embeddings: {e}")
            
        return embeddings
    
    def _validate_patient_ids(self, candidate_ids: List[str]) -> List[str]:
        """Only keep patient IDs that exist in all three data sources"""
        clinical_ids = set(self.clinical_data['record_id'].astype(str))
        embedding_ids = set(self.patient_embeddings.keys())
        site_ids = set(self.site_data['patient_id'].astype(str))
        
        valid_ids = []
        for pid in candidate_ids:
            if pid in clinical_ids and pid in embedding_ids and pid in site_ids:
                valid_ids.append(pid)
        
        self.logger.info(f"Validated {len(valid_ids)}/{len(candidate_ids)} patient IDs")
        return valid_ids
    
    def _find_common_patient_ids(self) -> List[str]:
        """Find patients with data in all modalities"""
        clinical_ids = set(self.clinical_data['record_id'].astype(str))
        embedding_ids = set(self.patient_embeddings.keys())
        site_ids = set(self.site_data['patient_id'].astype(str))
        
        common_ids = list(clinical_ids & embedding_ids & site_ids)
        
        self.logger.info(f"Clinical IDs: {len(clinical_ids)}")
        self.logger.info(f"Embedding IDs: {len(embedding_ids)}")
        self.logger.info(f"Site IDs: {len(site_ids)}")
        self.logger.info(f"Common IDs: {len(common_ids)}")
        
        return common_ids
    
    def __len__(self) -> int:
        return len(self.patient_ids)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        pid = self.patient_ids[idx]
        
        try:
            # Get site data
            site_rows = self.site_data[self.site_data['patient_id'] == pid]
            if len(site_rows) == 0:
                raise ValueError(f"No site data found for patient {pid}")
            site_row = site_rows.iloc[0]
            
            # Get clinical features
            if pid not in self.clinical_features.index:
                self.logger.warning(f"Patient {pid} not found in clinical features, using zeros")
                clinical_vector = np.zeros(len(self.feature_order))
            else:
                clinical_vector = self.clinical_features.loc[pid].values
                # Final safety check for NaN values
                if np.isnan(clinical_vector).any():
                    self.logger.warning(f"NaN values found for patient {pid}, replacing with 0")
                    clinical_vector = np.nan_to_num(clinical_vector, nan=0.0)
            
            # Get patient embedding
            if pid not in self.patient_embeddings:
                raise ValueError(f"No embedding found for patient {pid}")
            patient_embedding = self.patient_embeddings[pid]
            
            # Extract pathology findings safely
            pathology_features = []
            pathology_names = ['a_lines_finding', 'b_lines_finding', 'small_consolidations_finding', 
                             'large_consolidations_finding', 'pleural_effusion_finding']
            
            for feature in pathology_names:
                value = site_row.get(feature, 0.0)
                if pd.isna(value):
                    value = 0.0
                pathology_features.append(float(value))
            
            return {
                'patient_id': pid,
                'clinical_features': torch.tensor(clinical_vector, dtype=torch.float32),
                'patient_embedding': patient_embedding,
                'tb_label': torch.tensor(float(site_row.get('tb_label', 0.0)), dtype=torch.float32),
                'tb_prob': torch.tensor(float(site_row.get('tb_prob', 0.0)), dtype=torch.float32),
                'pathology_findings': torch.tensor(pathology_features, dtype=torch.float32),
                'mil_attention': torch.tensor(float(site_row.get('mil_attention', 0.0)), dtype=torch.float32),
                'site_observation_count': torch.tensor(float(site_row.get('site_observation_count', 1)), dtype=torch.float32)
            }
            
        except Exception as e:
            self.logger.error(f"Error loading data for patient {pid}: {e}")
            
            # Return safe fallback data
            clinical_vector = np.zeros(len(self.feature_order))
            dummy_embedding = torch.zeros_like(list(self.patient_embeddings.values())[0]) if self.patient_embeddings else torch.zeros(512)
            
            return {
                'patient_id': pid,
                'clinical_features': torch.tensor(clinical_vector, dtype=torch.float32),
                'patient_embedding': dummy_embedding,
                'tb_label': torch.tensor(0.0, dtype=torch.float32),
                'tb_prob': torch.tensor(0.0, dtype=torch.float32),
                'pathology_findings': torch.zeros(5, dtype=torch.float32),
                'mil_attention': torch.tensor(0.0, dtype=torch.float32),
                'site_observation_count': torch.tensor(1.0, dtype=torch.float32)
            }

    def get_feature_info(self) -> Dict:
        """Return information needed by models.py"""
        return {
            'feature_schema': self.feature_schema,
            'feature_order': self.feature_order,
            'clinical_dim': self.clinical_dim,
            'num_patients': len(self.patient_ids)
        }


def load_fold_split_robust(fold_csv_path: str) -> Tuple[List[str], List[str], List[str]]:
    """Robust fold splitting function"""
    if not os.path.exists(fold_csv_path):
        raise FileNotFoundError(f"Fold split file not found: {fold_csv_path}")
    
    df = pd.read_csv(fold_csv_path)
    
    if 'split' in df.columns and 'patient_id' in df.columns:
        print("📋 Using 'split' column format")
        train_ids = df[df["split"] == "train"]["patient_id"].astype(str).tolist()
        val_ids = df[df["split"] == "val"]["patient_id"].astype(str).tolist()
        test_ids = df[df["split"] == "test"]["patient_id"].astype(str).tolist()
    elif all(col in df.columns for col in ['train_ids', 'valid_ids', 'test_ids']):
        print("📋 Using separate ID columns format")
        train_ids = df['train_ids'].dropna().astype(str).tolist()
        val_ids = df['valid_ids'].dropna().astype(str).tolist()
        test_ids = df['test_ids'].dropna().astype(str).tolist()
    else:
        raise ValueError(f"Unsupported CSV format. Columns: {list(df.columns)}")
    
    print(f"📊 Loaded splits - Train: {len(train_ids)}, Val: {len(val_ids)}, Test: {len(test_ids)}")
    
    return train_ids, val_ids, test_ids


def custom_collate_fn(batch: List[Dict[str, any]]) -> Dict[str, any]:
    """Custom collate function"""
    collated_batch = {
        'patient_id': [item['patient_id'] for item in batch],
        'clinical_features': torch.stack([item['clinical_features'] for item in batch]),
        'patient_embedding': torch.stack([item['patient_embedding'] for item in batch]),
        'tb_label': torch.stack([item['tb_label'] for item in batch]),
        'tb_prob': torch.stack([item['tb_prob'] for item in batch]),
        'pathology_findings': torch.stack([item['pathology_findings'] for item in batch]),
        'mil_attention': torch.stack([item['mil_attention'] for item in batch]),
        'site_observation_count': torch.stack([item['site_observation_count'] for item in batch])
    }
    
    return collated_batch


def create_dataloaders(clinical_data_path: str, base_h5_path: str, base_site_path: str,
                           fold_num: int, base_fold_path: str = None, batch_size: int = 32, 
                           num_workers: int = 4, **dataset_kwargs) -> Tuple:
    """Create dataloaders with clean medical features only"""
    
    paths = {
        'train': (f"{base_h5_path}/train_full_model_fold{fold_num}_complex_data.h5",
                  f"{base_site_path}/train_full_model_fold{fold_num}_sites.csv"),
        'val':   (f"{base_h5_path}/val_full_model_fold{fold_num}_complex_data.h5",
                  f"{base_site_path}/val_full_model_fold{fold_num}_sites.csv"),
        'test':  (f"{base_h5_path}/test_full_model_fold{fold_num}_complex_data.h5",
                  f"{base_site_path}/test_full_model_fold{fold_num}_sites.csv")
    }

    if base_fold_path:
        fold_csv_path = f'{base_fold_path}/Fold_{fold_num}.csv'
        train_ids, val_ids, test_ids = load_fold_split_robust(fold_csv_path)
    else:
        train_ids = val_ids = test_ids = None

    # Create datasets with clean medical features
    train_dataset = MultimodalTBDataset(
        clinical_data_path, *paths['train'], 
        patient_ids=train_ids, fit_scalers=True, **dataset_kwargs
    )
    
    val_dataset = MultimodalTBDataset(
        clinical_data_path, *paths['val'], 
        patient_ids=val_ids, fit_scalers=False,
        clinical_scaler=train_dataset.clinical_scaler, **dataset_kwargs
    )
    
    test_dataset = MultimodalTBDataset(
        clinical_data_path, *paths['test'], 
        patient_ids=test_ids, fit_scalers=False,
        clinical_scaler=train_dataset.clinical_scaler, **dataset_kwargs
    )

    return (
        DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                  num_workers=num_workers, pin_memory=True, collate_fn=custom_collate_fn),
        DataLoader(val_dataset, batch_size=batch_size, shuffle=False, 
                  num_workers=num_workers, pin_memory=True, collate_fn=custom_collate_fn),
        DataLoader(test_dataset, batch_size=batch_size, shuffle=False, 
                  num_workers=num_workers, pin_memory=True, collate_fn=custom_collate_fn),
        train_dataset.clinical_scaler
    )


def analyze_dataset_distribution(dataset: MultimodalTBDataset) -> Dict:
    """Analyze clean medical dataset distribution"""
    
    if len(dataset) == 0:
        return {'error': 'Empty dataset'}
    
    try:
        sample_size = min(len(dataset), 100)
        tb_labels = []
        clinical_stats = []
        
        for i in range(sample_size):
            try:
                sample = dataset[i]
                tb_labels.append(sample['tb_label'].item())
                clinical_stats.append(sample['clinical_features'].numpy())
            except Exception as e:
                continue
        
        if not tb_labels:
            return {'error': 'No valid samples found'}
        
        tb_labels = np.array(tb_labels)
        clinical_stats = np.array(clinical_stats)
        
        analysis = {
            'total_patients': len(dataset),
            'sampled_patients': len(tb_labels),
            'tb_positive_cases': int(np.sum(tb_labels == 1)),
            'tb_negative_cases': int(np.sum(tb_labels == 0)),
            'tb_positive_ratio': float(np.mean(tb_labels == 1)),
            'clinical_features_dim': dataset.clinical_dim,
            'clinical_features_mean': float(clinical_stats.mean()),
            'clinical_features_std': float(clinical_stats.std()),
            'has_nan_values': bool(np.isnan(clinical_stats).any()),
            'feature_names_count': len(dataset.feature_order),
            'embedding_dim': list(dataset.patient_embeddings.values())[0].shape[0] if dataset.patient_embeddings else 'unknown',
            'feature_schema': {domain: len(features) for domain, features in dataset.feature_schema.items()}
        }
        
        return analysis
        
    except Exception as e:
        return {'error': f'Analysis failed: {str(e)}'}