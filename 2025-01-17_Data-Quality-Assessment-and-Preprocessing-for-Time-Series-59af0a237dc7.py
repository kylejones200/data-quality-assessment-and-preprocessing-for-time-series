# Description: Short example for Data Quality Assessment and Preprocessing for Time Series.



from datetime import datetime, timedelta
from scipy import stats
import logging
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)



# Create a sample time series with intentional quality issues
np.random.seed(42)
dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='D')
values = 100 + np.random.normal(0, 10, len(dates))

# Create DataFrame and introduce quality issues
df = pd.DataFrame({'timestamp': dates, 'value': values})
df.loc[10:15, 'value'] = np.nan  # Missing values
df.loc[50, 'value'] = 1000  # Obvious outlier
df = df.sample(frac=0.95)  # Create irregular timestamps

# Sort the dataframe by timestamp to ensure proper plotting
df = df.sort_values('timestamp')

def assess_time_series_completeness(df, timestamp_col, value_col):
    # Check for missing values
    missing_count = df[value_col].isna().sum()
    total_count = len(df)
    
    # Check for temporal gaps
    time_diff = df[timestamp_col].diff()
    expected_diff = pd.Timedelta(time_diff.mode()[0])
    irregular_intervals = time_diff != expected_diff
    
    # Generate completeness report
    logger.info(f"Data Completeness Assessment:")
    logger.info(f"Total observations: {total_count}")
    logger.info(f"Missing values: {missing_count} ({missing_count/total_count:.2%})")
    logger.info(f"Irregular intervals: {irregular_intervals.sum()}")
    
    return {
        'missing_ratio': missing_count/total_count,
        'irregular_intervals': irregular_intervals.sum()
    }

def detect_anomalies(df, value_col, n_std=3):
    # Calculate rolling statistics
    rolling_mean = df[value_col].rolling(window=7, center=True).mean()
    rolling_std = df[value_col].rolling(window=7, center=True).std()
    
    # Define bounds for anomaly detection
    upper_bound = rolling_mean + (n_std * rolling_std)
    lower_bound = rolling_mean - (n_std * rolling_std)
    
    # Identify anomalies
    anomalies = (df[value_col] > upper_bound) | (df[value_col] < lower_bound)
    
    return anomalies, upper_bound, lower_bound

# Implement anomaly detection and visualization

anomalies, upper, lower = detect_anomalies(df, 'value')
plt.figure(figsize=(12, 6))
plt.plot(df['timestamp'], df['value'], label='Original')
plt.plot(df['timestamp'], upper, 'r--', label='Upper Bound')
plt.plot(df['timestamp'], lower, 'r--', label='Lower Bound')
plt.scatter(df[anomalies]['timestamp'], df[anomalies]['value'], 
            color='red', label='Anomalies')
plt.title('Time Series Anomaly Detection')
plt.legend()
plt.show()

def preprocess_time_series(df, timestamp_col, value_col, target_freq='D'):
    # Sort by timestamp
    df = df.sort_values(timestamp_col)
    
    # Create regular time index
    full_idx = pd.date_range(
        start=df[timestamp_col].min(),
        end=df[timestamp_col].max(),
        freq=target_freq
    )
    
    # Reindex and interpolate
    df_regular = df.set_index(timestamp_col).reindex(full_idx)
    
    # Handle missing values with multiple methods
    df_regular['linear_interpolation'] = df_regular[value_col].interpolate(method='linear')
    df_regular['forward_fill'] = df_regular[value_col].ffill()  # Changed from fillna(method='ffill')
    df_regular['backward_fill'] = df_regular[value_col].bfill()  # Changed from fillna(method='bfill')
    
    # Add rolling statistics
    df_regular['rolling_mean'] = df_regular[value_col].rolling(window=7, min_periods=1).mean()
    
    return df_regular

def validate_preprocessing(original_df, processed_df, value_col):
    # Compare basic statistics
    original_stats = original_df[value_col].describe()
    processed_stats = processed_df['linear_interpolation'].describe()
    
    # Check for remaining missing values
    remaining_missing = processed_df['linear_interpolation'].isna().sum()
    
    # Assess distribution similarity
    ks_statistic, p_value = stats.ks_2samp(
        original_df[value_col].dropna(),
        processed_df['linear_interpolation'].dropna()
    )
    
    logger.info("Validation Results:")
    logger.info(f"Remaining missing values: {remaining_missing}")
    logger.info(f"Distribution similarity test p-value: {p_value:.4f}")
    
    return {
        'original_stats': original_stats,
        'processed_stats': processed_stats,
        'ks_test_p_value': p_value
    }

# Run the analysis
completeness_metrics = assess_time_series_completeness(df, 'timestamp', 'value')
anomalies, upper, lower = detect_anomalies(df, 'value')
preprocessed_df = preprocess_time_series(df, 'timestamp', 'value')
validation_results = validate_preprocessing(df, preprocessed_df, 'value')

# Create and save visualizations with improved formatting
# 1. Anomaly Detection Plot
plt.figure(figsize=(15, 7))
plt.plot(df['timestamp'], df['value'], label='Original', alpha=0.7)
plt.plot(df['timestamp'], upper, 'r--', label='Upper Bound', alpha=0.5)
plt.plot(df['timestamp'], lower, 'r--', label='Lower Bound', alpha=0.5)
plt.scatter(df[anomalies]['timestamp'], df[anomalies]['value'], 
            color='red', label='Anomalies', s=100)
plt.title('Time Series Anomaly Detection', fontsize=14)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Value', fontsize=12)
plt.legend(fontsize=10)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('anomaly_detection.png', dpi=300, bbox_inches='tight')
plt.close()

# 2. Interpolation Methods Comparison Plot
plt.figure(figsize=(15, 7))
plt.plot(preprocessed_df.index, preprocessed_df['value'], 'o', label='Original', alpha=0.3, markersize=4)
plt.plot(preprocessed_df.index, preprocessed_df['linear_interpolation'], label='Linear Interpolation', linewidth=2)
plt.plot(preprocessed_df.index, preprocessed_df['forward_fill'], label='Forward Fill', linewidth=2)
plt.plot(preprocessed_df.index, preprocessed_df['rolling_mean'], label='Rolling Mean', linewidth=2)
plt.title('Comparison of Different Interpolation Methods', fontsize=14)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Value', fontsize=12)
plt.legend(fontsize=10)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('interpolation_methods.png', dpi=300, bbox_inches='tight')
plt.close()

# 3. Data Distribution Plot
plt.figure(figsize=(15, 6))
plt.subplot(1, 2, 1)
df['value'].hist(bins=30, alpha=0.7, color='skyblue')
plt.title('Original Data Distribution', fontsize=14)
plt.xlabel('Value', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.subplot(1, 2, 2)
preprocessed_df['linear_interpolation'].hist(bins=30, alpha=0.7, color='lightgreen')
plt.title('Processed Data Distribution', fontsize=14)
plt.xlabel('Value', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.tight_layout()
plt.savefig('data_distribution.png', dpi=300, bbox_inches='tight')
plt.close()

logger.info("Analysis complete. Plots have been saved as:")
logger.info("- anomaly_detection.png")
logger.info("- interpolation_methods.png")
logger.info("- data_distribution.png")
