import pandas as pd
import numpy as np
import joblib
import logging
import traceback
import os
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class SimplePropertyPredictor:
    def __init__(self):
        self.model = RandomForestRegressor(
            n_estimators=150, 
            max_depth=20, 
            random_state=42, 
            n_jobs=-1,
            min_samples_split=5,
            min_samples_leaf=2
        )
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_columns = None
        self.data_quality_report = {}
        self.training_metrics = {}
        
    def validate_data_quality(self, df):
        """Validate data quality and log comprehensive issues"""
        logger.info("Starting comprehensive data quality validation...")
        
        quality_report = {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'missing_values': {},
            'invalid_values': {},
            'outliers': {},
            'data_types': {},
            'duplicates': 0,
            'memory_usage_mb': 0
        }
        
        try:
            # Memory usage
            quality_report['memory_usage_mb'] = df.memory_usage(deep=True).sum() / (1024 * 1024)
            logger.info(f"DataFrame memory usage: {quality_report['memory_usage_mb']:.2f} MB")
            
            # Check for duplicates
            quality_report['duplicates'] = df.duplicated().sum()
            if quality_report['duplicates'] > 0:
                logger.warning(f"Found {quality_report['duplicates']} duplicate rows")
            
            # Check missing values
            missing_counts = df.isnull().sum()
            quality_report['missing_values'] = missing_counts[missing_counts > 0].to_dict()
            
            if quality_report['missing_values']:
                logger.warning(f"Missing values found: {quality_report['missing_values']}")
                for col, count in quality_report['missing_values'].items():
                    percentage = (count / len(df)) * 100
                    logger.warning(f"  {col}: {count} missing ({percentage:.2f}%)")
            
            # Check for invalid values (inf, -inf)
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            for col in numeric_columns:
                if col in df.columns:
                    invalid_count = df[col].isin([np.inf, -np.inf]).sum()
                    if invalid_count > 0:
                        quality_report['invalid_values'][col] = invalid_count
                        logger.warning(f"Found {invalid_count} infinite values in {col}")
            
            # Check for outliers (values beyond 3 standard deviations)
            outlier_columns = ['current_value', 'square_feet', 'lot_size', 'bedrooms', 'bathrooms']
            for col in outlier_columns:
                if col in df.columns and df[col].dtype in ['int64', 'float64']:
                    try:
                        mean_val = df[col].mean()
                        std_val = df[col].std()
                        if std_val > 0:  # Avoid division by zero
                            outliers = df[(df[col] < mean_val - 3*std_val) | (df[col] > mean_val + 3*std_val)]
                            if len(outliers) > 0:
                                quality_report['outliers'][col] = {
                                    'count': len(outliers),
                                    'percentage': (len(outliers) / len(df)) * 100,
                                    'min_outlier': outliers[col].min(),
                                    'max_outlier': outliers[col].max()
                                }
                                logger.info(f"Found {len(outliers)} outliers in {col} ({(len(outliers)/len(df)*100):.2f}%)")
                    except Exception as e:
                        logger.warning(f"Error checking outliers for {col}: {str(e)}")
            
            # Data type validation
            quality_report['data_types'] = {col: str(dtype) for col, dtype in df.dtypes.items()}
            
            # Check for suspicious values
            if 'current_value' in df.columns:
                zero_values = (df['current_value'] <= 0).sum()
                if zero_values > 0:
                    logger.warning(f"Found {zero_values} properties with zero or negative values")
                    
                extreme_high = (df['current_value'] > 10000000).sum()  # >$10M
                if extreme_high > 0:
                    logger.info(f"Found {extreme_high} properties valued over $10M")
            
            if 'year_built' in df.columns:
                future_built = (df['year_built'] > 2025).sum()
                old_built = (df['year_built'] < 1800).sum()
                if future_built > 0:
                    logger.warning(f"Found {future_built} properties built in the future")
                if old_built > 0:
                    logger.warning(f"Found {old_built} properties built before 1800")
            
            self.data_quality_report = quality_report
            logger.info(f"Data quality validation completed. Summary:")
            logger.info(f"  Total rows: {quality_report['total_rows']:,}")
            logger.info(f"  Total columns: {quality_report['total_columns']}")
            logger.info(f"  Memory usage: {quality_report['memory_usage_mb']:.2f} MB")
            logger.info(f"  Duplicates: {quality_report['duplicates']}")
            logger.info(f"  Columns with missing values: {len(quality_report['missing_values'])}")
            logger.info(f"  Columns with outliers: {len(quality_report['outliers'])}")
            
        except Exception as e:
            logger.error(f"Error during data quality validation: {str(e)}")
            logger.error(traceback.format_exc())
            
        return quality_report

    def load_and_preprocess_data(self, csv_file, sample_size=500000):
        """Load and preprocess the property dataset with comprehensive error handling and data loss tracking"""
        logger.info(f"Loading dataset from {csv_file}...")
        
        # Data loss tracking
        data_loss_tracker = {
            'initial_rows': 0,
            'after_loading': 0,
            'after_null_removal': 0,
            'after_value_filter': 0,
            'after_sqft_filter': 0,
            'after_feature_engineering': 0,
            'after_outlier_removal': 0,
            'final_rows': 0,
            'loss_reasons': {}
        }
        
        try:
            # Check if file exists and get file info
            if not os.path.exists(csv_file):
                raise FileNotFoundError(f"Dataset file {csv_file} not found")
            
            file_size_mb = os.path.getsize(csv_file) / (1024 * 1024)
            logger.info(f"File size: {file_size_mb:.2f} MB")
            
            # Estimate total rows
            with open(csv_file, 'r') as f:
                first_line = f.readline()
                estimated_total_rows = file_size_mb * 1024 * 1024 / len(first_line)
                logger.info(f"Estimated total rows in file: {estimated_total_rows:,.0f}")
            
            # Read dataset with progressive fallback
            try:
                logger.info(f"Attempting to load {sample_size:,} rows...")
                df = pd.read_csv(csv_file, nrows=sample_size)
                data_loss_tracker['initial_rows'] = int(min(sample_size, estimated_total_rows))
                data_loss_tracker['after_loading'] = len(df)
                logger.info(f"Dataset loaded successfully: {len(df):,} rows, {len(df.columns)} columns")
            except pd.errors.EmptyDataError:
                logger.error("CSV file is empty")
                raise
            except pd.errors.ParserError as e:
                logger.error(f"Error parsing CSV file: {str(e)}")
                logger.info("Attempting to load with error handling...")
                df = pd.read_csv(csv_file, nrows=sample_size, error_bad_lines=False, warn_bad_lines=True)
                data_loss_tracker['after_loading'] = len(df)
                logger.warning(f"Loaded with some rows skipped due to parsing errors")
            except MemoryError:
                logger.warning(f"Memory error with sample_size={sample_size:,}, reducing to 200,000")
                sample_size = 200000
                df = pd.read_csv(csv_file, nrows=sample_size)
                data_loss_tracker['initial_rows'] = sample_size
                data_loss_tracker['after_loading'] = len(df)
                logger.info(f"Dataset loaded with reduced size: {len(df):,} rows")
            
            # Log column information
            logger.info(f"Available columns: {list(df.columns)}")
            
            # Validate data quality
            self.validate_data_quality(df)
            
            # Check for required columns
            required_columns = ['current_value', 'year_built', 'total_5yr_appreciation']
            optional_columns = ['square_feet', 'bedrooms', 'bathrooms', 'state', 'property_type']
            
            missing_required = [col for col in required_columns if col not in df.columns]
            missing_optional = [col for col in optional_columns if col not in df.columns]
            
            if missing_required:
                logger.error(f"Missing required columns: {missing_required}")
                raise ValueError(f"Dataset missing required columns: {missing_required}")
            
            if missing_optional:
                logger.warning(f"Missing optional columns (will use defaults): {missing_optional}")
            
            # Data cleaning and preprocessing with detailed tracking
            logger.info("Starting comprehensive data preprocessing...")
            
            # 1. Remove duplicate rows
            initial_for_duplicates = len(df)
            df = df.drop_duplicates()
            duplicates_removed = initial_for_duplicates - len(df)
            if duplicates_removed > 0:
                logger.warning(f"Removed {duplicates_removed:,} duplicate rows")
                data_loss_tracker['loss_reasons']['duplicates'] = duplicates_removed
            
            # 2. Remove rows with null required columns
            before_null_removal = len(df)
            df = df.dropna(subset=required_columns)
            data_loss_tracker['after_null_removal'] = len(df)
            null_removed = before_null_removal - len(df)
            if null_removed > 0:
                logger.warning(f"Removed {null_removed:,} rows with null required values")
                data_loss_tracker['loss_reasons']['null_required'] = null_removed
            
            # 3. Remove invalid property values (negative, zero, or unrealistic)
            before_value_filter = len(df)
            df = df[
                (df['current_value'] > 0) & 
                (df['current_value'] < 50000000) &  # Less than $50M
                (df['total_5yr_appreciation'] > -df['current_value'])  # Can't lose more than 100%
            ]
            data_loss_tracker['after_value_filter'] = len(df)
            value_filtered = before_value_filter - len(df)
            if value_filtered > 0:
                logger.warning(f"Removed {value_filtered:,} rows with invalid property values")
                data_loss_tracker['loss_reasons']['invalid_values'] = value_filtered
            
            # 4. Handle square footage if available
            if 'square_feet' in df.columns:
                before_sqft_filter = len(df)
                df = df[(df['square_feet'] >= 300) & (df['square_feet'] <= 20000)]
                data_loss_tracker['after_sqft_filter'] = len(df)
                sqft_filtered = before_sqft_filter - len(df)
                if sqft_filtered > 0:
                    logger.warning(f"Removed {sqft_filtered:,} rows with unrealistic square footage")
                    data_loss_tracker['loss_reasons']['sqft_filter'] = sqft_filtered
            else:
                # Create default square footage
                logger.info("Square footage not available, creating estimated values")
                df['square_feet'] = np.random.randint(1000, 3000, len(df))
                data_loss_tracker['after_sqft_filter'] = len(df)
            
            # 5. Handle other missing columns with defaults
            if 'bedrooms' not in df.columns:
                df['bedrooms'] = np.random.randint(1, 5, len(df))
                logger.info("Created default bedroom values")
                
            if 'bathrooms' not in df.columns:
                df['bathrooms'] = np.random.uniform(1, 3.5, len(df)).round(1)
                logger.info("Created default bathroom values")
                
            if 'state' not in df.columns:
                states = ['CA', 'TX', 'FL', 'NY', 'PA', 'IL', 'OH', 'GA', 'NC', 'MI']
                df['state'] = np.random.choice(states, len(df))
                logger.info("Created default state values")
                
            if 'property_type' not in df.columns:
                types = ['Single Family Home', 'Townhouse', 'Condo', 'Ranch', 'Colonial']
                df['property_type'] = np.random.choice(types, len(df))
                logger.info("Created default property type values")
            
            # 6. Feature engineering with error handling
            before_feature_eng = len(df)
            try:
                df['property_age'] = 2025 - df['year_built']
                df['price_per_sqft'] = df['current_value'] / df['square_feet']
                df['value_per_bedroom'] = df['current_value'] / df['bedrooms'].replace(0, 1)
                df['appreciation_rate'] = df['total_5yr_appreciation'] / df['current_value']
                
                # Remove infinite values from engineered features
                df = df.replace([np.inf, -np.inf], np.nan)
                df = df.dropna(subset=['price_per_sqft', 'value_per_bedroom', 'appreciation_rate'])
                
                data_loss_tracker['after_feature_engineering'] = len(df)
                feature_eng_removed = before_feature_eng - len(df)
                if feature_eng_removed > 0:
                    logger.warning(f"Removed {feature_eng_removed:,} rows during feature engineering")
                    data_loss_tracker['loss_reasons']['feature_engineering'] = feature_eng_removed
                    
            except Exception as e:
                logger.error(f"Error in feature engineering: {str(e)}")
                raise
            
            # 7. Remove extreme outliers (optional, aggressive cleaning)
            before_outlier_removal = len(df)
            
            # Remove extreme outliers using IQR method for key features
            for column in ['current_value', 'price_per_sqft']:
                if column in df.columns:
                    Q1 = df[column].quantile(0.01)  # More conservative than 0.25
                    Q3 = df[column].quantile(0.99)  # More conservative than 0.75
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 3 * IQR  # Very conservative
                    upper_bound = Q3 + 3 * IQR
                    
                    outliers_before = len(df)
                    df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
                    outliers_removed = outliers_before - len(df)
                    
                    if outliers_removed > 0:
                        logger.info(f"Removed {outliers_removed:,} extreme outliers from {column}")
            
            data_loss_tracker['after_outlier_removal'] = len(df)
            outlier_removed = before_outlier_removal - len(df)
            if outlier_removed > 0:
                data_loss_tracker['loss_reasons']['outliers'] = outlier_removed
            
            # 8. Fill remaining missing values with intelligent defaults
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            missing_before_fill = df[numeric_columns].isnull().sum().sum()
            if missing_before_fill > 0:
                logger.info(f"Filling {missing_before_fill:,} remaining missing numeric values")
                
                # Use median for most columns, mode for categorical-like columns
                for col in numeric_columns:
                    if df[col].isnull().sum() > 0:
                        if col in ['bedrooms', 'bathrooms']:
                            fill_value = df[col].mode().iloc[0] if not df[col].mode().empty else df[col].median()
                        else:
                            fill_value = df[col].median()
                        df[col] = df[col].fillna(fill_value)
                        logger.debug(f"Filled {col} missing values with {fill_value}")
            
            # Final data summary
            data_loss_tracker['final_rows'] = len(df)
            total_loss = data_loss_tracker['initial_rows'] - data_loss_tracker['final_rows']
            loss_percentage = (total_loss / data_loss_tracker['initial_rows']) * 100 if data_loss_tracker['initial_rows'] > 0 else 0
            
            # Comprehensive logging
            logger.info("="*80)
            logger.info("DATA PREPROCESSING SUMMARY")
            logger.info("="*80)
            logger.info(f"Initial estimated rows: {data_loss_tracker['initial_rows']:,}")
            logger.info(f"Successfully loaded: {data_loss_tracker['after_loading']:,}")
            logger.info(f"After null removal: {data_loss_tracker['after_null_removal']:,}")
            logger.info(f"After value filtering: {data_loss_tracker['after_value_filter']:,}")
            logger.info(f"After sqft filtering: {data_loss_tracker['after_sqft_filter']:,}")
            logger.info(f"After feature engineering: {data_loss_tracker['after_feature_engineering']:,}")
            logger.info(f"After outlier removal: {data_loss_tracker['after_outlier_removal']:,}")
            logger.info(f"Final rows: {data_loss_tracker['final_rows']:,}")
            logger.info(f"Total data loss: {total_loss:,} rows ({loss_percentage:.2f}%)")
            
            if data_loss_tracker['loss_reasons']:
                logger.info("\nData loss breakdown:")
                for reason, count in data_loss_tracker['loss_reasons'].items():
                    percentage = (count / data_loss_tracker['initial_rows']) * 100
                    logger.info(f"  {reason}: {count:,} rows ({percentage:.2f}%)")
            
            # Quality checks
            if loss_percentage > 30:
                logger.warning(f"HIGH DATA LOSS: {loss_percentage:.2f}%. Consider reviewing data quality or preprocessing steps.")
            elif loss_percentage > 15:
                logger.warning(f"Moderate data loss: {loss_percentage:.2f}%. Data quality may need attention.")
            else:
                logger.info(f"Acceptable data loss: {loss_percentage:.2f}%")
            
            if data_loss_tracker['final_rows'] < 5000:
                logger.error("INSUFFICIENT DATA: Less than 5,000 rows remaining. Model performance may be poor.")
                if data_loss_tracker['final_rows'] < 1000:
                    raise ValueError("Critical: Less than 1,000 rows remaining. Cannot train reliable model.")
            
            # Final data validation
            logger.info(f"\nFinal dataset statistics:")
            logger.info(f"  Rows: {len(df):,}")
            logger.info(f"  Columns: {len(df.columns)}")
            logger.info(f"  Memory usage: {df.memory_usage(deep=True).sum() / (1024*1024):.2f} MB")
            logger.info(f"  Current value range: ${df['current_value'].min():,.0f} - ${df['current_value'].max():,.0f}")
            logger.info(f"  Average appreciation: ${df['total_5yr_appreciation'].mean():,.0f}")
            logger.info("="*80)
            
            return df
            
        except Exception as e:
            logger.error(f"CRITICAL ERROR in load_and_preprocess_data: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def prepare_features(self, df):
        """Prepare features for training with comprehensive error handling"""
        logger.info("Preparing features for model training...")
        
        try:
            # Create a copy to avoid modifying original data
            df_processed = df.copy()
            
            # Categorical features to encode
            categorical_features = ['state', 'property_type', 'school_district_rating']
            
            # Encode categorical features with robust error handling
            for col in categorical_features:
                if col in df_processed.columns:
                    try:
                        if col not in self.label_encoders:
                            # First time encoding - fit the encoder
                            self.label_encoders[col] = LabelEncoder()
                            df_processed[col + '_encoded'] = self.label_encoders[col].fit_transform(df_processed[col].astype(str))
                            logger.info(f"Encoded categorical feature: {col} ({len(self.label_encoders[col].classes_)} classes)")
                        else:
                            # Subsequent encoding - handle unseen categories
                            known_classes = set(self.label_encoders[col].classes_)
                            df_col_values = set(df_processed[col].astype(str).unique())
                            unseen_values = df_col_values - known_classes
                            
                            if unseen_values:
                                logger.warning(f"Found {len(unseen_values)} unseen categories in {col}: {list(unseen_values)[:5]}...")
                                # Replace unseen values with most common class
                                most_common = self.label_encoders[col].classes_[0]
                                df_processed[col] = df_processed[col].astype(str).replace(list(unseen_values), most_common)
                                logger.info(f"Replaced unseen values with: {most_common}")
                            
                            df_processed[col + '_encoded'] = self.label_encoders[col].transform(df_processed[col].astype(str))
                    except Exception as e:
                        logger.error(f"Error encoding {col}: {str(e)}")
                        # Create dummy encoding if original fails
                        df_processed[col + '_encoded'] = 0
                        logger.warning(f"Created dummy encoding for {col}")
                else:
                    logger.warning(f"Categorical feature {col} not found in dataset, creating default")
                    df_processed[col + '_encoded'] = 0
            
            # Define numerical features
            numerical_features = [
                'current_value', 'year_built', 'bedrooms', 'bathrooms', 'square_feet',
                'lot_size', 'property_tax_annual', 'hoa_monthly', 'monthly_rent_estimate',
                'property_age', 'price_per_sqft', 'value_per_bedroom', 'appreciation_rate'
            ]
            
            # Check which numerical features are available
            available_numerical = [col for col in numerical_features if col in df_processed.columns]
            missing_numerical = [col for col in numerical_features if col not in df_processed.columns]
            
            if missing_numerical:
                logger.warning(f"Missing numerical features: {missing_numerical}")
                # Create default values for missing features
                for col in missing_numerical:
                    if col == 'lot_size':
                        df_processed[col] = 0.25  # Default quarter acre
                    elif col == 'property_tax_annual':
                        df_processed[col] = df_processed['current_value'] * 0.012  # 1.2% default
                    elif col == 'hoa_monthly':
                        df_processed[col] = 0  # No HOA by default
                    elif col == 'monthly_rent_estimate':
                        df_processed[col] = df_processed['current_value'] * 0.008  # 0.8% of value monthly
                    else:
                        df_processed[col] = 0
                    logger.info(f"Created default values for missing feature: {col}")
            
            logger.info(f"Using {len(available_numerical)} numerical features (including defaults)")
            
            # Combine all features
            categorical_encoded = [col + '_encoded' for col in categorical_features]
            all_available_numerical = [col for col in numerical_features if col in df_processed.columns]
            feature_cols = all_available_numerical + categorical_encoded
            
            logger.info(f"Total features for model: {len(feature_cols)}")
            logger.info(f"Feature list: {feature_cols}")
            
            # Create feature matrix
            X = df_processed[feature_cols].copy()
            
            # Comprehensive data cleaning
            initial_rows = len(X)
            
            # Replace infinite values
            inf_counts = np.isinf(X.select_dtypes(include=[np.number])).sum().sum()
            if inf_counts > 0:
                logger.warning(f"Found {inf_counts} infinite values, replacing with NaN")
                X = X.replace([np.inf, -np.inf], np.nan)
            
            # Handle missing values intelligently
            missing_count = X.isnull().sum().sum()
            if missing_count > 0:
                logger.info(f"Filling {missing_count} missing feature values")
                
                # Fill with median for numerical, mode for categorical
                for col in X.columns:
                    if X[col].isnull().sum() > 0:
                        if col.endswith('_encoded'):
                            # Categorical encoded features - use mode
                            fill_value = X[col].mode().iloc[0] if not X[col].mode().empty else 0
                        else:
                            # Numerical features - use median
                            fill_value = X[col].median() if not X[col].isna().all() else 0
                        
                        X[col] = X[col].fillna(fill_value)
                        missing_filled = X[col].isnull().sum()
                        if missing_filled == 0:
                            logger.debug(f"Successfully filled missing values in {col} with {fill_value}")
            
            # Final validation
            remaining_nulls = X.isnull().sum().sum()
            if remaining_nulls > 0:
                logger.error(f"Still have {remaining_nulls} missing values after cleaning")
                # Final fallback - fill with 0
                X = X.fillna(0)
                logger.warning("Used zero-fill as final fallback for missing values")
            
            # Check for any remaining data quality issues
            if not np.isfinite(X.select_dtypes(include=[np.number])).all().all():
                logger.error("Data still contains non-finite values after cleaning")
                X = X.replace([np.inf, -np.inf, np.nan], 0)
                logger.warning("Replaced all non-finite values with zero as final fallback")
            
            # Store feature columns for prediction
            self.feature_columns = X.columns.tolist()
            
            final_rows = len(X)
            if final_rows != initial_rows:
                logger.warning(f"Lost {initial_rows - final_rows} rows during feature preparation")
            
            logger.info(f"Feature preparation completed successfully!")
            logger.info(f"  Final shape: {X.shape}")
            logger.info(f"  Data types: {X.dtypes.value_counts().to_dict()}")
            logger.info(f"  Memory usage: {X.memory_usage(deep=True).sum() / (1024*1024):.2f} MB")
            
            return X
            
        except Exception as e:
            logger.error(f"CRITICAL ERROR in prepare_features: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def train_model(self, csv_file):
        """Train the model with comprehensive validation and performance analysis"""
        training_start_time = datetime.now()
        
        try:
            logger.info("="*80)
            logger.info("STARTING COMPREHENSIVE MODEL TRAINING")
            logger.info("="*80)
            
            # Load and preprocess data
            df = self.load_and_preprocess_data(csv_file, sample_size=500000)
            
            # Prepare features
            X = self.prepare_features(df)
            
            # Define target variable
            y = df['current_value'] + df['total_5yr_appreciation']
            
            logger.info(f"\nTraining dataset summary:")
            logger.info(f"  Samples: {X.shape[0]:,}")
            logger.info(f"  Features: {X.shape[1]}")
            logger.info(f"  Target variable (5-year future value) statistics:")
            logger.info(f"    Mean: ${y.mean():,.2f}")
            logger.info(f"    Median: ${y.median():,.2f}")
            logger.info(f"    Std: ${y.std():,.2f}")
            logger.info(f"    Min: ${y.min():,.2f}")
            logger.info(f"    Max: ${y.max():,.2f}")
            logger.info(f"    25th percentile: ${y.quantile(0.25):,.2f}")
            logger.info(f"    75th percentile: ${y.quantile(0.75):,.2f}")
            
            # Enhanced data split (70% train, 30% test)
            logger.info(f"\nSplitting data: 70% train, 30% test...")
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=42, shuffle=True
            )
            
            logger.info(f"Data split completed:")
            logger.info(f"  Training set: {X_train.shape[0]:,} samples ({X_train.shape[0]/len(X)*100:.1f}%)")
            logger.info(f"  Test set: {X_test.shape[0]:,} samples ({X_test.shape[0]/len(X)*100:.1f}%)")
            logger.info(f"  Feature dimensions: {X_train.shape[1]}")
            
            # Feature scaling
            logger.info(f"\nScaling features using StandardScaler...")
            scaling_start = datetime.now()
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            scaling_time = (datetime.now() - scaling_start).total_seconds()
            logger.info(f"Feature scaling completed in {scaling_time:.2f} seconds")
            
            # Log feature scaling statistics
            logger.info(f"Feature scaling statistics:")
            logger.info(f"  Mean of scaled training features: {X_train_scaled.mean():.4f}")
            logger.info(f"  Std of scaled training features: {X_train_scaled.std():.4f}")
            
            # Train model
            logger.info(f"\nTraining Random Forest model...")
            logger.info(f"Model parameters:")
            logger.info(f"  n_estimators: {self.model.n_estimators}")
            logger.info(f"  max_depth: {self.model.max_depth}")
            logger.info(f"  min_samples_split: {self.model.min_samples_split}")
            logger.info(f"  min_samples_leaf: {self.model.min_samples_leaf}")
            
            model_training_start = datetime.now()
            self.model.fit(X_train_scaled, y_train)
            model_training_time = (datetime.now() - model_training_start).total_seconds()
            
            logger.info(f"Model training completed in {model_training_time:.2f} seconds!")
            
            # Comprehensive model evaluation
            logger.info(f"\nEvaluating model performance...")
            
            # Training set predictions
            y_train_pred = self.model.predict(X_train_scaled)
            train_mae = mean_absolute_error(y_train, y_train_pred)
            train_mse = mean_squared_error(y_train, y_train_pred)
            train_rmse = np.sqrt(train_mse)
            train_r2 = r2_score(y_train, y_train_pred)
            
            # Test set predictions
            y_test_pred = self.model.predict(X_test_scaled)
            test_mae = mean_absolute_error(y_test, y_test_pred)
            test_mse = mean_squared_error(y_test, y_test_pred)
            test_rmse = np.sqrt(test_mse)
            test_r2 = r2_score(y_test, y_test_pred)
            
            # Additional metrics
            def mean_absolute_percentage_error(y_true, y_pred):
                return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
            
            def median_absolute_error(y_true, y_pred):
                return np.median(np.abs(y_true - y_pred))
            
            train_mape = mean_absolute_percentage_error(y_train, y_train_pred)
            test_mape = mean_absolute_percentage_error(y_test, y_test_pred)
            train_median_ae = median_absolute_error(y_train, y_train_pred)
            test_median_ae = median_absolute_error(y_test, y_test_pred)
            
            # Prediction accuracy analysis
            def accuracy_within_range(y_true, y_pred, percentage):
                errors = np.abs(y_true - y_pred) / y_true
                return (errors <= percentage/100).mean() * 100
            
            train_acc_5pct = accuracy_within_range(y_train, y_train_pred, 5)
            train_acc_10pct = accuracy_within_range(y_train, y_train_pred, 10)
            train_acc_15pct = accuracy_within_range(y_train, y_train_pred, 15)
            
            test_acc_5pct = accuracy_within_range(y_test, y_test_pred, 5)
            test_acc_10pct = accuracy_within_range(y_test, y_test_pred, 10)
            test_acc_15pct = accuracy_within_range(y_test, y_test_pred, 15)
            
            # Overfitting analysis
            r2_difference = train_r2 - test_r2
            mae_increase_pct = ((test_mae - train_mae) / train_mae) * 100
            rmse_increase_pct = ((test_rmse - train_rmse) / train_rmse) * 100
            
            # Performance summary
            total_training_time = (datetime.now() - training_start_time).total_seconds()
            
            logger.info("="*80)
            logger.info("COMPREHENSIVE MODEL PERFORMANCE ANALYSIS")
            logger.info("="*80)
            
            logger.info("TRAINING SET PERFORMANCE:")
            logger.info(f"  Mean Absolute Error (MAE): ${train_mae:,.2f}")
            logger.info(f"  Root Mean Square Error (RMSE): ${train_rmse:,.2f}")
            logger.info(f"  Median Absolute Error: ${train_median_ae:,.2f}")
            logger.info(f"  R² Score: {train_r2:.6f}")
            logger.info(f"  Mean Absolute Percentage Error (MAPE): {train_mape:.2f}%")
            logger.info(f"  Predictions within 5%: {train_acc_5pct:.1f}%")
            logger.info(f"  Predictions within 10%: {train_acc_10pct:.1f}%")
            logger.info(f"  Predictions within 15%: {train_acc_15pct:.1f}%")
            
            logger.info("\nTEST SET PERFORMANCE:")
            logger.info(f"  Mean Absolute Error (MAE): ${test_mae:,.2f}")
            logger.info(f"  Root Mean Square Error (RMSE): ${test_rmse:,.2f}")
            logger.info(f"  Median Absolute Error: ${test_median_ae:,.2f}")
            logger.info(f"  R² Score: {test_r2:.6f}")
            logger.info(f"  Mean Absolute Percentage Error (MAPE): {test_mape:.2f}%")
            logger.info(f"  Predictions within 5%: {test_acc_5pct:.1f}%")
            logger.info(f"  Predictions within 10%: {test_acc_10pct:.1f}%")
            logger.info(f"  Predictions within 15%: {test_acc_15pct:.1f}%")
            
            logger.info(f"\nOVERFITTING ANALYSIS:")
            logger.info(f"  R² Difference (Train - Test): {r2_difference:.6f}")
            logger.info(f"  MAE Increase (Test vs Train): {mae_increase_pct:.2f}%")
            logger.info(f"  RMSE Increase (Test vs Train): {rmse_increase_pct:.2f}%")
            
            # Overfitting warnings and recommendations
            if r2_difference > 0.1:
                logger.warning("⚠️  SIGNIFICANT OVERFITTING: Large R² difference (>0.1)")
                logger.warning("   Recommendations: Reduce max_depth, increase min_samples_split")
            elif r2_difference > 0.05:
                logger.warning("⚠️  MODERATE OVERFITTING: R² difference >0.05")
                logger.warning("   Recommendations: Consider regularization or early stopping")
            else:
                logger.info("✅ Overfitting levels are acceptable")
            
            if mae_increase_pct > 25:
                logger.warning("⚠️  HIGH TEST ERROR INCREASE: MAE increased >25%")
            elif mae_increase_pct > 15:
                logger.warning("⚠️  MODERATE TEST ERROR INCREASE: MAE increased >15%")
            else:
                logger.info("✅ Test error increase is acceptable")
            
            # Feature importance analysis
            if hasattr(self.model, 'feature_importances_'):
                logger.info(f"\nFEATURE IMPORTANCE ANALYSIS:")
                feature_importance = pd.DataFrame({
                    'feature': self.feature_columns,
                    'importance': self.model.feature_importances_
                }).sort_values('importance', ascending=False)
                
                logger.info("Top 15 Most Important Features:")
                for idx, row in feature_importance.head(15).iterrows():
                    logger.info(f"  {row['feature']:<25}: {row['importance']:.6f}")
                
                # Feature importance insights
                top_5_importance = feature_importance.head(5)['importance'].sum()
                logger.info(f"\nTop 5 features contribute {top_5_importance:.1%} of total importance")
            
            # Performance quality assessment
            logger.info(f"\nMODEL QUALITY ASSESSMENT:")
            
            quality_score = 0
            assessments = []
            
            # R² Score assessment
            if test_r2 >= 0.95:
                assessments.append("✅ Excellent R² score (≥0.95)")
                quality_score += 25
            elif test_r2 >= 0.90:
                assessments.append("✅ Very good R² score (≥0.90)")
                quality_score += 20
            elif test_r2 >= 0.85:
                assessments.append("⚠️ Good R² score (≥0.85)")
                quality_score += 15
            else:
                assessments.append("❌ Poor R² score (<0.85)")
                quality_score += 0
            
            # MAPE assessment
            if test_mape <= 5:
                assessments.append("✅ Excellent MAPE (≤5%)")
                quality_score += 25
            elif test_mape <= 10:
                assessments.append("✅ Very good MAPE (≤10%)")
                quality_score += 20
            elif test_mape <= 15:
                assessments.append("⚠️ Acceptable MAPE (≤15%)")
                quality_score += 15
            else:
                assessments.append("❌ Poor MAPE (>15%)")
                quality_score += 0
            
            # Accuracy within 10% assessment
            if test_acc_10pct >= 80:
                assessments.append("✅ Excellent accuracy within 10% (≥80%)")
                quality_score += 25
            elif test_acc_10pct >= 70:
                assessments.append("✅ Good accuracy within 10% (≥70%)")
                quality_score += 20
            elif test_acc_10pct >= 60:
                assessments.append("⚠️ Moderate accuracy within 10% (≥60%)")
                quality_score += 15
            else:
                assessments.append("❌ Poor accuracy within 10% (<60%)")
                quality_score += 0
            
            # Overfitting assessment
            if r2_difference <= 0.02 and mae_increase_pct <= 10:
                assessments.append("✅ Minimal overfitting")
                quality_score += 25
            elif r2_difference <= 0.05 and mae_increase_pct <= 20:
                assessments.append("✅ Low overfitting")
                quality_score += 20
            else:
                assessments.append("⚠️ Noticeable overfitting")
                quality_score += 10
            
            for assessment in assessments:
                logger.info(f"  {assessment}")
            
            overall_grade = "A+" if quality_score >= 90 else "A" if quality_score >= 80 else "B" if quality_score >= 70 else "C" if quality_score >= 60 else "D"
            logger.info(f"\nOVERALL MODEL GRADE: {overall_grade} (Score: {quality_score}/100)")
            
            # Timing summary
            logger.info(f"\nTIMING SUMMARY:")
            logger.info(f"  Data loading & preprocessing: {(model_training_start - training_start_time).total_seconds():.2f} seconds")
            logger.info(f"  Feature scaling: {scaling_time:.2f} seconds")
            logger.info(f"  Model training: {model_training_time:.2f} seconds")
            logger.info(f"  Total training time: {total_training_time:.2f} seconds")
            
            logger.info("="*80)
            
            # Store comprehensive training metrics
            self.training_metrics = {
                'train_metrics': {
                    'mae': train_mae,
                    'rmse': train_rmse,
                    'r2': train_r2,
                    'mape': train_mape,
                    'median_ae': train_median_ae,
                    'accuracy_5pct': train_acc_5pct,
                    'accuracy_10pct': train_acc_10pct,
                    'accuracy_15pct': train_acc_15pct
                },
                'test_metrics': {
                    'mae': test_mae,
                    'rmse': test_rmse,
                    'r2': test_r2,
                    'mape': test_mape,
                    'median_ae': test_median_ae,
                    'accuracy_5pct': test_acc_5pct,
                    'accuracy_10pct': test_acc_10pct,
                    'accuracy_15pct': test_acc_15pct
                },
                'overfitting_analysis': {
                    'r2_difference': r2_difference,
                    'mae_increase_pct': mae_increase_pct,
                    'rmse_increase_pct': rmse_increase_pct
                },
                'data_info': {
                    'train_samples': X_train.shape[0],
                    'test_samples': X_test.shape[0],
                    'features': X.shape[1],
                    'data_quality_report': self.data_quality_report
                },
                'performance_assessment': {
                    'quality_score': quality_score,
                    'grade': overall_grade,
                    'assessments': assessments
                },
                'timing': {
                    'total_time': total_training_time,
                    'training_time': model_training_time,
                    'scaling_time': scaling_time
                }
            }
            
            return self.training_metrics
            
        except Exception as e:
            logger.error(f"CRITICAL ERROR in train_model: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def predict_future_value(self, property_data):
        """Predict future property value with comprehensive error handling and confidence estimation"""
        try:
            logger.info("Making property value prediction...")
            
            # Convert input to DataFrame if needed
            if isinstance(property_data, dict):
                df = pd.DataFrame([property_data])
            else:
                df = property_data.copy()
            
            # Validate required fields
            required_fields = ['current_value', 'year_built', 'square_feet', 'bedrooms']
            missing_fields = [field for field in required_fields if field not in df.columns or df[field].isnull().any()]
            
            if missing_fields:
                logger.error(f"Missing required fields for prediction: {missing_fields}")
                raise ValueError(f"Missing required fields: {missing_fields}")
            
            # Validate data ranges
            if df['current_value'].iloc[0] <= 0:
                raise ValueError("Current value must be positive")
            if df['year_built'].iloc[0] < 1800 or df['year_built'].iloc[0] > 2025:
                raise ValueError("Year built must be between 1800 and 2025")
            if df['square_feet'].iloc[0] < 300 or df['square_feet'].iloc[0] > 20000:
                raise ValueError("Square feet must be between 300 and 20,000")
            
            # Add default values for missing optional fields
            defaults = {
                'bathrooms': 2.0,
                'state': 'CA',
                'property_type': 'Single Family Home',
                'school_district_rating': 'Good',
                'lot_size': 0.25,
                'property_tax_annual': df['current_value'].iloc[0] * 0.012,
                'hoa_monthly': 0,
                'monthly_rent_estimate': df['current_value'].iloc[0] * 0.008
            }
            
            for col, default_val in defaults.items():
                if col not in df.columns:
                    df[col] = default_val
                elif df[col].isnull().any():
                    df[col] = df[col].fillna(default_val)
            
            # Engineer features
            try:
                df['property_age'] = 2025 - df['year_built']
                df['price_per_sqft'] = df['current_value'] / df['square_feet']
                df['value_per_bedroom'] = df['current_value'] / df['bedrooms'].replace(0, 1)
                df['total_5yr_appreciation'] = 0  # Placeholder for feature preparation
                df['appreciation_rate'] = 0  # Placeholder
                
                logger.info("Feature engineering completed for prediction")
            except Exception as e:
                logger.error(f"Error in feature engineering for prediction: {str(e)}")
                raise
            
            # Prepare features using the same pipeline as training
            X = self.prepare_features(df)
            
            # Ensure all required features are present
            missing_features = [col for col in self.feature_columns if col not in X.columns]
            if missing_features:
                logger.warning(f"Missing features: {missing_features}, filling with zeros")
                for col in missing_features:
                    X[col] = 0
            
            # Reorder columns to match training
            X = X[self.feature_columns]
            
            # Scale features
            X_scaled = self.scaler.transform(X)
            
            # Make prediction
            predicted_future_value = self.model.predict(X_scaled)[0]
            
            # Calculate additional insights
            current_value = df['current_value'].iloc[0]
            predicted_appreciation = predicted_future_value - current_value
            appreciation_percentage = (predicted_appreciation / current_value) * 100
            annual_appreciation = appreciation_percentage / 5  # 5-year period
            
            # Estimate prediction confidence based on training metrics
            test_mae = self.training_metrics.get('test_metrics', {}).get('mae', 50000)
            confidence_interval = test_mae * 1.96  # 95% confidence interval approximation
            
            prediction_result = {
                'current_value': current_value,
                'predicted_future_value': predicted_future_value,
                'predicted_appreciation': predicted_appreciation,
                'appreciation_percentage': appreciation_percentage,
                'annual_appreciation_rate': annual_appreciation,
                'confidence_interval': confidence_interval,
                'prediction_range': {
                    'lower': predicted_future_value - confidence_interval,
                    'upper': predicted_future_value + confidence_interval
                }
            }
            
            logger.info(f"Prediction completed:")
            logger.info(f"  Current value: ${current_value:,.2f}")
            logger.info(f"  Predicted 5-year value: ${predicted_future_value:,.2f}")
            logger.info(f"  Total appreciation: ${predicted_appreciation:,.2f} ({appreciation_percentage:.2f}%)")
            logger.info(f"  Annual appreciation rate: {annual_appreciation:.2f}%")
            logger.info(f"  95% Confidence interval: ±${confidence_interval:,.2f}")
            
            return prediction_result
            
        except Exception as e:
            logger.error(f"Error in predict_future_value: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def save_model(self, model_path='enhanced_property_model.joblib'):
        """Save the trained model with comprehensive metadata"""
        try:
            model_data = {
                'model': self.model,
                'scaler': self.scaler,
                'label_encoders': self.label_encoders,
                'feature_columns': self.feature_columns,
                'training_metrics': self.training_metrics,
                'data_quality_report': self.data_quality_report,
                'model_version': '2.0',
                'training_date': datetime.now().isoformat(),
                'model_parameters': self.model.get_params()
            }
            
            joblib.dump(model_data, model_path)
            logger.info(f"Enhanced model saved successfully to {model_path}")
            logger.info(f"Model file size: {os.path.getsize(model_path) / (1024*1024):.2f} MB")
            
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            raise

    def load_model(self, model_path='enhanced_property_model.joblib'):
        """Load a previously trained model with validation"""
        try:
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file {model_path} not found")
            
            model_data = joblib.load(model_path)
            
            # Load model components
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.label_encoders = model_data['label_encoders']
            self.feature_columns = model_data['feature_columns']
            self.training_metrics = model_data.get('training_metrics', {})
            self.data_quality_report = model_data.get('data_quality_report', {})
            
            logger.info(f"Enhanced model loaded successfully from {model_path}")
            logger.info(f"Model version: {model_data.get('model_version', 'Unknown')}")
            logger.info(f"Training date: {model_data.get('training_date', 'Unknown')}")
            logger.info(f"Features: {len(self.feature_columns)}")
            
            if self.training_metrics:
                test_r2 = self.training_metrics.get('test_metrics', {}).get('r2', 'Unknown')
                test_mae = self.training_metrics.get('test_metrics', {}).get('mae', 'Unknown')
                logger.info(f"Model performance - R²: {test_r2}, MAE: ${test_mae:,.2f}" if test_mae != 'Unknown' else f"Model performance - R²: {test_r2}")
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise

def main():
    """Main function with comprehensive error handling and execution flow"""
    try:
        logger.info("="*100)
        logger.info("ENHANCED PROPERTY VALUE PREDICTION MODEL TRAINING")
        logger.info("="*100)
        
        # Initialize predictor
        predictor = SimplePropertyPredictor()
        
        # Check for dataset
        dataset_files = [
            'property_dataset_3gb.csv',
            'property_dataset_300mb.csv',
            'property_dataset.csv'
        ]
        
        dataset_file = None
        for file in dataset_files:
            if os.path.exists(file):
                dataset_file = file
                break
        
        if not dataset_file:
            logger.error("No dataset file found. Please ensure one of these files exists:")
            for file in dataset_files:
                logger.error(f"  - {file}")
            raise FileNotFoundError("Dataset file not found")
        
        logger.info(f"Using dataset: {dataset_file}")
        
        # Train model
        training_metrics = predictor.train_model(dataset_file)
        
        # Save model
        model_filename = 'enhanced_property_model.joblib'
        predictor.save_model(model_filename)
        
        # Test prediction
        logger.info("\nTesting model with sample prediction...")
        sample_property = {
            'current_value': 500000,
            'year_built': 2010,
            'square_feet': 2000,
            'bedrooms': 3,
            'bathrooms': 2.5,
            'state': 'CA',
            'property_type': 'Single Family Home'
        }
        
        prediction = predictor.predict_future_value(sample_property)
        
        logger.info("Sample prediction completed:")
        logger.info(f"  Input: {sample_property}")
        logger.info(f"  Predicted 5-year value: ${prediction['predicted_future_value']:,.2f}")
        logger.info(f"  Expected appreciation: {prediction['appreciation_percentage']:.2f}%")
        
        logger.info("="*100)
        logger.info("TRAINING COMPLETED SUCCESSFULLY!")
        logger.info("="*100)
        
        # Print final summary
        test_metrics = training_metrics.get('test_metrics', {})
        logger.info("FINAL MODEL SUMMARY:")
        logger.info(f"  Model file: {model_filename}")
        logger.info(f"  R² Score: {test_metrics.get('r2', 'N/A'):.4f}")
        logger.info(f"  MAE: ${test_metrics.get('mae', 0):,.2f}")
        logger.info(f"  MAPE: {test_metrics.get('mape', 0):.2f}%")
        logger.info(f"  Grade: {training_metrics.get('performance_assessment', {}).get('grade', 'N/A')}")
        
        return True
        
    except Exception as e:
        logger.error("="*100)
        logger.error("TRAINING FAILED!")
        logger.error("="*100)
        logger.error(f"Error: {str(e)}")
        logger.error(traceback.format_exc())
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎉 Model training completed successfully!")
        print("📁 Model saved as 'enhanced_property_model.joblib'")
        print("📊 Check 'training.log' for detailed training logs")
    else:
        print("\n❌ Model training failed!")
        print("📋 Check 'training.log' for error details")
