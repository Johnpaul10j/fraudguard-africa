import joblib
import pandas as pd
import numpy as np
from datetime import datetime

# Load model and features
model = joblib.load('models/fraudguard_xgboost_full.pkl')
feature_cols = joblib.load('models/feature_cols.pkl')

def predict_fraud(transaction_data):
    """
    Make real-time fraud prediction
    transaction_data: dict containing transaction features
    """
    # Convert input to DataFrame
    df = pd.DataFrame([transaction_data])
    
    # Apply same feature engineering
    df['balance_diff_orig'] = df['oldbalanceOrg'] - df['newbalanceOrig']
    df['balance_diff_dest'] = df['oldbalanceDest'] - df['newbalanceDest']
    df['amount_to_oldbalance_ratio'] = df['amount'] / (df['oldbalanceOrg'] + 1e-8)
    df['amount_to_newbalance_ratio'] = df['amount'] / (df['newbalanceOrig'] + 1e-8)
    
    df['hour'] = df['step'] % 24
    df['is_night'] = df['hour'].isin([0,1,2,3,4,5,22,23]).astype(int)
    
    # One-hot encoding for type
    df = pd.get_dummies(df, columns=['type'], prefix='type', drop_first=True)
    
    # Handle missing columns (if some transaction types are not present)
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0
    
    # Reorder columns to match training
    df = df[feature_cols]
    
    # Make prediction
    probability = model.predict_proba(df)[0][1]
    prediction = model.predict(df)[0]
    
    risk_level = "High" if probability > 0.7 else "Medium" if probability > 0.3 else "Low"
    
    return {
        "is_fraud": bool(prediction),
        "fraud_probability": float(probability),
        "risk_level": risk_level,
        "timestamp": datetime.now().isoformat()
    }


# Test the function
if __name__ == "__main__":
    # Example transaction
    sample_transaction = {
        'step': 150,
        'type': 'CASH_OUT',
        'amount': 250000.0,
        'nameOrig': 'C123456789',
        'oldbalanceOrg': 300000.0,
        'newbalanceOrig': 50000.0,
        'nameDest': 'C987654321',
        'oldbalanceDest': 10000.0,
        'newbalanceDest': 260000.0
    }
    
    result = predict_fraud(sample_transaction)
    print(result)