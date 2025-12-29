import pandas as pd
import numpy as np
import os
def inject_drift(df, feature, drift_magnitude, drift_type='shift'):
    """
    Injects drift into a specified feature of the dataframe.

    Parameters:
    - df: pandas DataFrame
    - feature: str, the feature/column name to inject drift into
    - drift_magnitude: float, magnitude of the drift
    - drift_type: str, type of drift ('shift' or 'scaling')

    Returns:
    - df_drifted: pandas DataFrame with injected drift
    """
    df_drifted = df.copy()
    
    if drift_type == 'shift':
        df_drifted[feature] += drift_magnitude
    elif drift_type == 'scale':
        mean = df_drifted[feature].mean()
        df_drifted[feature] = mean + (df_drifted[feature] - mean) * drift_magnitude
    else:
        raise ValueError("Unsupported drift_type. Use 'shift' or 'scale'.")
    
    return df_drifted

def generate_batches(reference_df, n_batches=12, drift_config=None):
    """
    Generates batches of data with injected drift.

    Parameters:
    - reference_df: pandas DataFrame, the reference dataset
    - n_batches: int, number of batches to generate
    - drift_config: dict, configuration for drift injection

    Returns:
    - batches: list of pandas DataFrames with injected drift
    """
    batches = []
    n_samples = len(reference_df) // n_batches
    
    for i in range(n_batches):
        # Sample from reference (simulating new production data)
        batch = reference_df.sample(n=n_samples, replace=True, random_state=i)
        batch = batch.reset_index(drop=True)
        
        if drift_config and i > 3:  # Drift starts after batch 3
            drift_intensity = (i - 3) / (n_batches - 3)  # 0 to 1
            
            for feature, config in drift_config.items():
                magnitude = config['max_drift'] * drift_intensity
                batch = inject_drift(batch, feature, magnitude, config['type'])
        
        batch['batch_id'] = i
        batches.append(batch)
    
    return pd.concat(batches, ignore_index=True)


if __name__ == '__main__':
    import os
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.join(script_dir, '..')
    
    reference_path = os.path.join(project_root, 'data', 'processed', 'reference_with_labels.csv')
    output_path = os.path.join(project_root, 'data', 'processed', 'simulated_batches.csv')
    
    reference_df = pd.read_csv(reference_path)
    # Define which features will drift, picked the 3 most important features
    drift_config = {
        'PAY_0': {'type': 'shift', 'max_drift': 3},
        'BILL_AMT1': {'type': 'shift', 'max_drift': 50000},
        'PAY_AMT2': {'type': 'scale', 'max_drift': 2}
         
        
    }

    # Generate 12 batches (think of them as 12 weeks of production data)
    batches_df = generate_batches(reference_df, n_batches=20, drift_config=drift_config)

    batches_df.to_csv(output_path, index=False)
    print(f"Generated {batches_df['batch_id'].nunique()} batches")
    print(f"Total samples: {len(batches_df)}")
    print(f"\nDrift applied to: {list(drift_config.keys())}")