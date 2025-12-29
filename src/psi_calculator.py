import pandas as pd
import numpy as np
import os

def calculate_psi(reference, current , bins=10):
    breakpoints = np.percentile(reference, np.linspace(0, 100, bins + 1))
    #handle duplicate breakpoints
    breakpoints= np.unique(breakpoints)
    ref_counts, _ = np.histogram(reference, bins=breakpoints)
    curr_counts, _ = np.histogram(current, bins=breakpoints)    
    epsilo = 1e-10
    ref_percents = ref_counts / len(reference) + epsilo
    curr_percents = curr_counts / len(current) + epsilo

    #calculate PSI
    psi = np.sum((curr_percents - ref_percents) * np.log(curr_percents / ref_percents))
    return psi

def calculate_psi_for_batch(reference_df, batch_df, features):
    psi_scores={}
    for feature in features:
        psi = calculate_psi(reference_df[feature].values, batch_df[feature].values)
        psi_scores[feature] = psi
    return psi_scores

if __name__ == '__main__':
# Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Build paths relative to the script location
    reference_path = os.path.join(script_dir, '..', 'data', 'processed', 'reference_data.csv')
    output_path = os.path.join(script_dir, '..', 'data', 'processed', 'simulated_batches.csv')
    output = os.path.join(script_dir, '..', 'data', 'processed', 'psi_results.csv')

    

    # Load reference data
    reference_df = pd.read_csv(reference_path)
    batches_df = pd.read_csv(output_path)

    # Features to monitor (focus on drifted ones + a few stable ones)
    monitor_features = ['PAY_0', 'BILL_AMT1', 'PAY_AMT2', 'AGE']  # AGE as a stable control

    # Calculate PSI for each batch
    results = []
    
    for batch_id in sorted(batches_df['batch_id'].unique()):
        batch = batches_df[batches_df['batch_id'] == batch_id]
        
        psi_scores = calculate_psi_for_batch(reference_df, batch, monitor_features)
        psi_scores['batch_id'] = batch_id
        results.append(psi_scores)
    
    # Create results dataframe
    psi_df = pd.DataFrame(results)
    psi_df = psi_df[['batch_id'] + monitor_features]  # Reorder columns
    
    print("\nPSI Scores by Batch:")
    print(psi_df.round(3).to_string(index=False))
    
    print("\n--- Interpretation ---")
    print("PSI < 0.1  : No significant drift")
    print("PSI 0.1-0.25: Moderate drift - investigate")
    print("PSI > 0.25 : Significant drift - model unreliable")
    
    psi_df.to_csv(output, index=False)
    print("\nSaved to psi_results.csv")