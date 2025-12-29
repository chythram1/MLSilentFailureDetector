# batch_evaluator.py
import pandas as pd
import numpy as np
import joblib
import os

def evaluate_batches(batches_df, model, scaler, target_col='default payment next month'):
    """
    Calculate accuracy for each batch.
    """
    results = []
    
    for batch_id in sorted(batches_df['batch_id'].unique()):
        batch = batches_df[batches_df['batch_id'] == batch_id].copy()
        
        # Separate features and target
        X_batch = batch.drop(columns=['batch_id', target_col])
        y_batch = batch[target_col]
        
        # Scale and predict
        X_scaled = scaler.transform(X_batch)
        y_pred = model.predict(X_scaled)
        y_prob = model.predict_proba(X_scaled)[:, 1]
        
        # Calculate metrics
        accuracy = (y_pred == y_batch).mean()
        avg_confidence = y_prob.mean()
        
        results.append({
            'batch_id': batch_id,
            'accuracy': accuracy,
            'avg_confidence': avg_confidence,
            'n_samples': len(batch)
        })
    
    return pd.DataFrame(results)


if __name__ == '__main__':
    # Setup paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.join(script_dir, '..')
    
    # Load model and scaler
    model = joblib.load(os.path.join(project_root, 'models', 'baseline_model.pkl'))
    scaler = joblib.load(os.path.join(project_root, 'models', 'scaler.pkl'))
    
    # Load batches - need to add target column back
    batches_df = pd.read_csv(os.path.join(project_root, 'data', 'processed', 'simulated_batches.csv'))
    
    # Load PSI results
    psi_df = pd.read_csv(os.path.join(project_root, 'data', 'processed', 'psi_results.csv'))
    
    # Evaluate
    accuracy_df = evaluate_batches(batches_df, model, scaler)
    
    # Merge PSI and accuracy
    combined_df = psi_df.merge(accuracy_df, on='batch_id')
    
    print("\n===== DRIFT vs ACCURACY =====\n")
    print(combined_df[['batch_id', 'PAY_0', 'BILL_AMT1', 'PAY_AMT2', 'accuracy']].round(3).to_string(index=False))
    
    # Save combined results
    output_path = os.path.join(project_root, 'data', 'processed', 'drift_accuracy_results.csv')
    combined_df.to_csv(output_path, index=False)
    print(f"\nSaved to {output_path}")