import pandas as pd
import numpy as np
import logging
from typing import List, Dict, Optional, Union
from sklearn.impute import SimpleImputer
from pathlib import Path

class RealisticDataSimulator:
    """
    Simulates data that preserves original statistical properties.
    Use this for creating realistic synthetic data that mimics your original dataset.
    """
    
    def __init__(self, random_state: int = 42, logger: Optional[logging.Logger] = None):
        """
        Initialize the RealisticDataSimulator.
        
        Parameters:
        - random_state: int, random seed for reproducibility
        - logger: logging.Logger, optional logger for tracking
        """
        self.random_state = random_state
        self.logger = logger or self._setup_default_logger()
        np.random.seed(random_state)
        
        # Storing configuration
        self.feature_types = {}
        self.sample_stats = {}
        self.target_name = None
        
    def _setup_default_logger(self) -> logging.Logger:
        """Setup default logger if none provided."""
        logger = logging.getLogger('RealisticDataSimulator')
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
    
    def load_and_clean_data(self, data_path: str = "data/raw/telco_churn.csv") -> pd.DataFrame:
        """
        Load and clean the dataset using your provided cleaning pipeline.
        """
        self.logger.info(f"Loading data from: {data_path}")
        
        # Checking if file exists
        if not Path(data_path).exists():
            raise FileNotFoundError(f"Data file not found at: {data_path}")
        
        # Loading data
        data = pd.read_csv(data_path)
        self.logger.info(f"Original data shape: {data.shape}")
        
        # cleaning pipeline
        drop_cols = ["Unnamed: 0", "X", "customer", "traintest", "churndep"] 
        df = data.copy()
        df = df.drop(columns=[c for c in drop_cols if c in df.columns])
        self.logger.info(f"After dropping columns: {df.shape}")
        
        # Identifying numeric and categorical columns
        continuous_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
        
        self.logger.info(f"Found {len(continuous_cols)} continuous columns")
        self.logger.info(f"Found {len(categorical_cols)} categorical columns")
        
        # Handling continuous columns - filter out completely NaN columns
        valid_continuous_cols = [col for col in continuous_cols if not df[col].isna().all()]
        
        if valid_continuous_cols:
            num_imputer = SimpleImputer(strategy='mean')
            df[valid_continuous_cols] = num_imputer.fit_transform(df[valid_continuous_cols])
            self.logger.info(f"Imputed continuous columns: {len(valid_continuous_cols)}")
        else:
            self.logger.warning("No valid continuous columns found for imputation")
        
        # Handling categorical columns
        if categorical_cols:
            cat_imputer = SimpleImputer(strategy='most_frequent')
            df[categorical_cols] = cat_imputer.fit_transform(df[categorical_cols])
            self.logger.info(f"Imputed categorical columns: {len(categorical_cols)}")
        else:
            self.logger.info("No categorical columns found")
        
        # Final checking for missing values
        missing_summary = df.isna().sum()
        if missing_summary.any():
            self.logger.warning(f"Missing values remain: {missing_summary[missing_summary > 0].to_dict()}")
        else:
            self.logger.info("All missing values filled successfully")
        
        self.logger.info(f"Final cleaned data shape: {df.shape}")
        return df
    
    def _get_representative_sample(self, df: pd.DataFrame, target: str, 
                                sample_size: int, min_samples_per_class: int = 100) -> pd.DataFrame:
        """
        Get a stratified sample ensuring class balance and category diversity.
        """
        self.logger.info(f"Getting representative sample of size {sample_size}")
        
        if len(df) < sample_size:
            self.logger.warning(f"Dataset smaller than requested sample size. Using all {len(df)} rows.")
            return df.copy()
        
        # Checking target distribution
        target_counts = df[target].value_counts()
        self.logger.info(f"Original target distribution: {target_counts.to_dict()}")
        
        # Ensuring we have multiple classes
        if len(target_counts) < 2:
            self.logger.warning("Target has only one class. Using random sampling.")
            return df.sample(n=min(sample_size, len(df)), random_state=self.random_state)
        
        # Stratified sampling by target
        samples_per_class = {}
        for class_val in target_counts.index:
            class_data = df[df[target] == class_val]
            n_samples = max(min_samples_per_class, 
                          int(sample_size * len(class_data) / len(df)))
            n_samples = min(n_samples, len(class_data))
            
            if len(class_data) > 0:
                samples_per_class[class_val] = class_data.sample(n=n_samples, random_state=self.random_state)
        
        # Combining stratified samples
        sampled_df = pd.concat(samples_per_class.values(), ignore_index=True)
        
        # Adding additional samples if needed
        if len(sampled_df) < sample_size:
            remaining = df.drop(sampled_df.index)
            additional_samples = remaining.sample(n=sample_size - len(sampled_df), 
                                               random_state=self.random_state)
            sampled_df = pd.concat([sampled_df, additional_samples], ignore_index=True)
        
        self.logger.info(f"Sampled target distribution: {sampled_df[target].value_counts().to_dict()}")
        
        # Validating categorical diversity
        categorical_cols = sampled_df.select_dtypes(exclude=[np.number]).columns.tolist()
        for col in categorical_cols[:5]:  # Check first 5
            unique_count = sampled_df[col].nunique()
            if unique_count <= 1:
                self.logger.warning(f"Column {col} has only {unique_count} unique value in sample")
        
        return sampled_df
    
    def _identify_feature_types(self, df: pd.DataFrame, target: str) -> Dict[str, List[str]]:
        """
        Identify continuous, categorical, and binary features.
        """
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        feature_numeric_cols = [col for col in numeric_cols if col != target]
        
        binary_cols = []
        continuous_cols = []
        
        for col in feature_numeric_cols:
            unique_vals = df[col].dropna().unique()
            if len(unique_vals) == 2:
                binary_cols.append(col)
            else:
                continuous_cols.append(col)
        
        categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
        all_categorical_cols = categorical_cols + binary_cols
        
        feature_types = {
            'continuous': continuous_cols,
            'categorical': categorical_cols,
            'binary': binary_cols,
            'all_categorical': all_categorical_cols
        }
        
        self.logger.info(f"Identified {len(continuous_cols)} continuous, "
                       f"{len(categorical_cols)} categorical, "
                       f"{len(binary_cols)} binary features")
        
        return feature_types
    
    def _simulate_continuous_features(self, df: pd.DataFrame, continuous_cols: List[str], 
                                   sample_size: int) -> pd.DataFrame:
        """
        Simulate continuous features preserving correlations and distributions.
        """
        if not continuous_cols:
            return pd.DataFrame()
            
        cont_data = df[continuous_cols]
        sim_df = pd.DataFrame()
        
        if len(cont_data) > len(continuous_cols) and len(continuous_cols) > 1:
            try:
                # Multivariate normal for correlation preservation
                mean_vec = cont_data.mean().values
                cov_matrix = cont_data.cov().values
                
                # Ensuring positive definite covariance
                min_eig = np.min(np.real(np.linalg.eigvals(cov_matrix)))
                if min_eig < 0:
                    cov_matrix -= 1.1 * min_eig * np.eye(*cov_matrix.shape)
                
                sim_data = np.random.multivariate_normal(mean_vec, cov_matrix, size=sample_size)
                sim_df = pd.DataFrame(sim_data, columns=continuous_cols)
                
                # Applying original value ranges and preserve distribution shape
                for col in continuous_cols:
                    min_val = cont_data[col].min()
                    max_val = cont_data[col].max()
                    # Adding some noise to prevent exact boundary values
                    sim_df[col] = sim_df[col].clip(min_val * 0.95, max_val * 1.05)
                    
                self.logger.info("Used multivariate normal for continuous feature simulation")
                
            except Exception as e:
                self.logger.warning(f"Multivariate simulation failed: {e}. Using independent sampling.")
                # Fallback to independent sampling with distribution preservation
                for col in continuous_cols:
                    # Sampling from empirical distribution
                    sim_df[col] = np.random.choice(cont_data[col], size=sample_size, replace=True)
        else:
            # Independent sampling preserving empirical distributions
            for col in continuous_cols:
                sim_df[col] = np.random.choice(cont_data[col], size=sample_size, replace=True)
            self.logger.info("Used independent sampling for continuous features")
        
        return sim_df
    
    def _simulate_categorical_features(self, df: pd.DataFrame, categorical_cols: List[str],
                                    original_df: pd.DataFrame, sample_size: int) -> pd.DataFrame:
        """
        Simulate categorical features preserving distributions and all categories.
        """
        sim_df = pd.DataFrame()
        
        for col in categorical_cols:
            clean_series = df[col].dropna()
            if len(clean_series) > 0:
                # Getting probability distribution from sample
                value_counts = clean_series.value_counts()
                
                # Preserving all original categories with their frequencies
                original_categories = original_df[col].dropna().unique()
                current_categories = value_counts.index
                missing_categories = set(original_categories) - set(current_categories)
                
                if missing_categories:
                    # Adding small probability for missing categories to maintain diversity
                    total_weight = 0.02 * len(missing_categories)  # 2% total for missing
                    remaining_weight = 1 - total_weight
                    
                    adjusted_probs = (value_counts / value_counts.sum()) * remaining_weight
                    for cat in missing_categories:
                        adjusted_probs[cat] = total_weight / len(missing_categories)
                    
                    probs = adjusted_probs
                else:
                    probs = value_counts / value_counts.sum()
                
                # Ensuring probabilities sum to 1
                probs = probs / probs.sum()
                
                sim_df[col] = np.random.choice(probs.index, size=sample_size, p=probs.values)
            else:
                # Fallback for empty columns
                sim_df[col] = np.random.choice([0, 1], size=sample_size)
        
        self.logger.info(f"Simulated {len(categorical_cols)} categorical features")
        return sim_df
    
    def fit(self, data_path: str = "data/raw/telco_churn.csv", target: str = "churn", 
            sample_size: int = 500) -> None:
        """
        Fit the simulator to the dataset.
        
        Parameters:
        - data_path: str, path to the data file
        - target: str, target column name
        - sample_size: int, size of representative sample to use for fitting
        """
        self.logger.info(f"Fitting RealisticDataSimulator with target '{target}'")
        self.target_name = target
        
        # Loading and clean data
        original_df = self.load_and_clean_data(data_path)
        
        # Getting representative sample
        self.representative_sample = self._get_representative_sample(original_df, target, sample_size)
        
        # Identifying feature types
        self.feature_types = self._identify_feature_types(self.representative_sample, target)
        
        # Store target distribution
        self.target_distribution = self.representative_sample[target].value_counts(normalize=True)
        
        # Storing original data for category preservation
        self.original_df = original_df
        
        # Storing descriptive statistics for validation
        self.original_stats = {
            'continuous_means': original_df[self.feature_types['continuous']].mean().to_dict(),
            'continuous_stds': original_df[self.feature_types['continuous']].std().to_dict(),
            'categorical_proportions': {},
            'target_distribution': original_df[target].value_counts(normalize=True).to_dict()
        }
        
        for col in self.feature_types['all_categorical']:
            self.original_stats['categorical_proportions'][col] = original_df[col].value_counts(normalize=True).to_dict()
        
        self.logger.info("RealisticDataSimulator fitting completed successfully")
    
    def simulate(self, n_samples: int = 1000) -> pd.DataFrame:
        """
        Generate synthetic data.
        
        Parameters:
        - n_samples: int, number of synthetic samples to generate
        
        Returns:
        - pandas DataFrame with synthetic data
        """
        if not hasattr(self, 'representative_sample'):
            raise ValueError("Simulator must be fitted before generating data. Call fit() first.")
        
        self.logger.info(f"Generating {n_samples} synthetic samples")
        
        # Simulating continuous features
        sim_continuous = self._simulate_continuous_features(
            self.representative_sample, self.feature_types['continuous'], n_samples
        )
        
        # Simulating categorical features
        sim_categorical = self._simulate_categorical_features(
            self.representative_sample, self.feature_types['all_categorical'], 
            self.original_df, n_samples
        )
        
        # Simulating target - ensuring both classes are present
        if len(self.target_distribution) < 2:
            # If sample has only one class, use original distribution
            target_probs = self.original_df[self.target_name].value_counts(normalize=True)
        else:
            target_probs = self.target_distribution
        
        sim_target = np.random.choice(target_probs.index, size=n_samples, p=target_probs.values)
        
        # Combining all features
        simulated_df = pd.concat([sim_continuous, sim_categorical], axis=1)
        simulated_df[self.target_name] = sim_target
        
        # Ensuring proper data types
        for col in self.feature_types['continuous']:
            if col in simulated_df.columns:
                simulated_df[col] = simulated_df[col].astype(float)
        
        for col in self.feature_types['all_categorical']:
            if col in simulated_df.columns:
                simulated_df[col] = simulated_df[col].astype(self.representative_sample[col].dtype)
        
        self._validate_simulation(simulated_df)
        
        self.logger.info(f"Successfully generated {len(simulated_df)} synthetic samples")
        return simulated_df
    
    def _validate_simulation(self, simulated_df: pd.DataFrame) -> None:
        """Validate the quality of simulated data against original statistics."""
        self.logger.info("Validating simulation quality...")
        
        # Target distribution preservation
        sim_target_dist = simulated_df[self.target_name].value_counts(normalize=True)
        target_diff = np.abs(
            pd.Series(self.original_stats['target_distribution']) - sim_target_dist
        ).sum()
        
        self.logger.info(f"Target distribution difference: {target_diff:.4f}")
        
        # Continuous feature statistics
        if self.feature_types['continuous']:
            for col in self.feature_types['continuous'][:3]:  # Check first 3
                orig_mean = self.original_stats['continuous_means'][col]
                sim_mean = simulated_df[col].mean()
                mean_diff = abs(orig_mean - sim_mean) / orig_mean if orig_mean != 0 else abs(orig_mean - sim_mean)
                
                self.logger.info(f"{col} - Mean difference: {mean_diff:.4f}")
        
        # Categorical diversity preservation
        for col in self.feature_types['all_categorical'][:3]:  # Check first 3
            orig_cats = self.original_df[col].nunique()
            sim_cats = simulated_df[col].nunique()
            cat_preservation = sim_cats / orig_cats if orig_cats > 0 else 1.0
            
            self.logger.info(f"{col} - Category preservation: {cat_preservation:.2%} ({sim_cats}/{orig_cats})")
        
        # Correlation preservation (if multiple continuous features)
        if len(self.feature_types['continuous']) >= 2:
            orig_corr = self.original_df[self.feature_types['continuous'][:2]].corr().iloc[0, 1]
            sim_corr = simulated_df[self.feature_types['continuous'][:2]].corr().iloc[0, 1]
            corr_diff = abs(orig_corr - sim_corr)
            self.logger.info(f"Correlation difference: {corr_diff:.4f}")
    
    def get_simulation_report(self) -> Dict:
        """Get comprehensive report about the simulation quality."""
        if not hasattr(self, 'original_stats'):
            raise ValueError("Simulator not fitted yet.")
        
        return {
            'target': self.target_name,
            'feature_types': self.feature_types,
            'original_stats': self.original_stats,
            'sample_size': len(self.representative_sample),
            'target_distribution': self.target_distribution.to_dict()
        }


class DriftedDataSimulator:
    """
    A robust dataset simulator with controlled drift injection for generating 
    synthetic data that can simulate real-world data drift scenarios.
    """
    
    def __init__(self, random_state: int = 42, logger: Optional[logging.Logger] = None):
        """
        Initialize the DriftedDataSimulator.
        
        Parameters:
        - random_state: int, random seed for reproducibility
        - logger: logging.Logger, optional logger for tracking
        """
        self.random_state = random_state
        self.logger = logger or self._setup_default_logger()
        np.random.seed(random_state)
        
        # Storing configuration
        self.feature_types = {}
        self.drift_config = {}
        
    def _setup_default_logger(self) -> logging.Logger:
        """Setup default logger if none provided."""
        logger = logging.getLogger('DriftedDataSimulator')
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
    
    def configure_drift(self, 
                       target_drift_strength: float = 0.3,
                       feature_drift_strength: float = 0.4,
                       categorical_drift_strength: float = 0.5,
                       correlation_drift_strength: float = 0.3,
                       introduce_new_categories: bool = True,
                       shift_marginal_distributions: bool = True,
                       extreme_value_injection: bool = True):
        """
        Configure the strength and type of data drift to inject.
        
        Parameters:
        - target_drift_strength: 0-1, how much to shift target distribution
        - feature_drift_strength: 0-1, how much to shift feature distributions
        - categorical_drift_strength: 0-1, how much to change category frequencies
        - correlation_drift_strength: 0-1, how much to change feature correlations
        - introduce_new_categories: whether to add new categories in categorical features
        - shift_marginal_distributions: whether to shift min/max/mean of continuous features
        - extreme_value_injection: whether to inject extreme values outside original ranges
        """
        self.drift_config = {
            'target_drift_strength': max(0, min(1, target_drift_strength)),
            'feature_drift_strength': max(0, min(1, feature_drift_strength)),
            'categorical_drift_strength': max(0, min(1, categorical_drift_strength)),
            'correlation_drift_strength': max(0, min(1, correlation_drift_strength)),
            'introduce_new_categories': introduce_new_categories,
            'shift_marginal_distributions': shift_marginal_distributions,
            'extreme_value_injection': extreme_value_injection
        }
        
        self.logger.info(f"Drift configuration set: {self.drift_config}")
    
    def load_and_clean_data(self, data_path: str = "data/raw/telco_churn.csv") -> pd.DataFrame:
        """Load and clean the dataset using your provided cleaning pipeline."""
        self.logger.info(f"Loading data from: {data_path}")

        # Checking if file exists
        if not Path(data_path).exists():
            raise FileNotFoundError(f"Data file not found at: {data_path}")

        # Loading data
        data = pd.read_csv(data_path)
        self.logger.info(f"Original data shape: {data.shape}")

        # cleaning pipeline
        drop_cols = ["Unnamed: 0", "X", "customer", "traintest", "churndep"] 
        df = data.copy()
        df = df.drop(columns=[c for c in drop_cols if c in df.columns])
        self.logger.info(f"After dropping columns: {df.shape}")

        # Identifying numeric and categorical columns
        continuous_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

        self.logger.info(f"Found {len(continuous_cols)} continuous columns")
        self.logger.info(f"Found {len(categorical_cols)} categorical columns")

        # Handling continuous columns - filter out completely NaN columns
        valid_continuous_cols = [col for col in continuous_cols if not df[col].isna().all()]

        if valid_continuous_cols:
            num_imputer = SimpleImputer(strategy='mean')
            df[valid_continuous_cols] = num_imputer.fit_transform(df[valid_continuous_cols])
            self.logger.info(f"Imputed continuous columns: {len(valid_continuous_cols)}")
        else:
            self.logger.warning("No valid continuous columns found for imputation")

        # Handling categorical columns
        if categorical_cols:
            cat_imputer = SimpleImputer(strategy='most_frequent')
            df[categorical_cols] = cat_imputer.fit_transform(df[categorical_cols])
            self.logger.info(f"Imputed categorical columns: {len(categorical_cols)}")
        else:
            self.logger.info("No categorical columns found")

        # Final checking for missing values
        missing_summary = df.isna().sum()
        if missing_summary.any():
            self.logger.warning(f"Missing values remain: {missing_summary[missing_summary > 0].to_dict()}")
        else:
            self.logger.info("All missing values filled successfully")

        self.logger.info(f"Final cleaned data shape: {df.shape}")
        return df
def main():
    """Main function to demonstrate data simulation."""
    import time
    
    print("=== Starting Data Simulation ===")
    
    # Initialize simulators
    realistic_simulator = RealisticDataSimulator(random_state=42)
    drifted_simulator = DriftedDataSimulator(random_state=42)
    
    try:
        # Realistic Data Simulation
        print("\n--- Realistic Data Simulation ---")
        start_time = time.time()
        
        realistic_simulator.fit(
            data_path="data/raw/telco_churn.csv", 
            target="churn", 
            sample_size=500
        )
        
        realistic_data = realistic_simulator.simulate(n_samples=1000)
        print(f"✓ Generated realistic data: {realistic_data.shape}")
        print(f"Target distribution:\n{realistic_data['churn'].value_counts()}")
        
        # Getting simulation report
        report = realistic_simulator.get_simulation_report()
        print(f"✓ Simulation completed in {time.time() - start_time:.2f} seconds")
        
        # Drifted Data Simulation  
        print("\n--- Drifted Data Simulation ---")
        start_time = time.time()
        
        # Configuring drift
        drifted_simulator.configure_drift(
            target_drift_strength=0.3,
            feature_drift_strength=0.4,
            categorical_drift_strength=0.5
        )
        
        # Note: You'll need to implement the fit and simulate methods for DriftedDataSimulator
        # drifted_data = drifted_simulator.simulate(n_samples=1000)
        # print(f"Generated drifted data: {drifted_data.shape}")
        
        print(f"Drift configuration completed in {time.time() - start_time:.2f} seconds")
        
        # Saving sample of generated data
        print("\n--- Saving Results ---")
        realistic_data.head(100).to_csv("data/processed/simulated_realistic_sample.csv", index=False)
        print("Saved realistic data sample to: data/processed/simulated_realistic_sample.csv")
        
        print(f"\n=== Data Simulation Completed Successfully ===")
        
    except Exception as e:
        print(f"Error during simulation: {e}")
        raise

if __name__ == "__main__":
        main()