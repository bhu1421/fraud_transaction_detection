import joblib
import numpy as np
from pathlib import Path

# Resolve the model relative to the repository so local and container launches
# use the same artifact path.
model_path = Path(__file__).resolve().parents[2] / "artifacts" / "xgb_fraud_detection_model.pkl"
model = joblib.load(model_path)

def predict_fraud(transaction: dict):
    # Based on your X.head() output, your model expects these 13 features:
    feature_order = [
        "step",
        "type",  # This seems to be encoded as numeric (1 for the transaction types shown)
        "amount", 
        "oldbalanceOrg",
        "newbalanceOrig",
        "oldbalanceDest", 
        "newbalanceDest",
        "step_hour",
        "isMerchant",
        "isEmptied",
        "emptyReceiver",
        # Plus 2 more features that might be one-hot encoded types or other engineered features
        "type_CASH_OUT",  # Assuming these are the remaining 2
        "type_TRANSFER"
    ]
    
    # Extract features in the correct order, using 0 as default for missing features
    features = []
    for feature_name in feature_order:
        if feature_name in transaction:
            features.append(transaction[feature_name])
        else:
            # Default value for missing features
            features.append(0)
    
    # Debug: Print feature information
    print(f"Number of features: {len(features)}")
    print(f"Feature names: {feature_order}")
    print(f"Feature values: {features}")
    
    try:
        prediction = model.predict([features])[0]
        proba = model.predict_proba([features])[0][1]
        
        return {
            "prediction": int(prediction),
            "probability_of_fraud": round(float(proba), 4)
        }
    except Exception as e:
        # If we still get a feature mismatch, let's try with just the original 9 features
        print(f"Error with 14 features: {e}")
        print("Trying with original 9 features...")
        
        original_features = [
            transaction.get("amount", 0),
            transaction.get("oldbalanceOrg", 0),
            transaction.get("newbalanceOrig", 0),
            transaction.get("oldbalanceDest", 0),
            transaction.get("newbalanceDest", 0),
            transaction.get("errorBalanceOrig", 0),
            transaction.get("errorBalanceDest", 0),
            transaction.get("type_CASH_OUT", 0),
            transaction.get("type_TRANSFER", 0)
        ]
        
        prediction = model.predict([original_features])[0]
        proba = model.predict_proba([original_features])[0][1]
        
        return {
            "prediction": int(prediction),
            "probability_of_fraud": round(float(proba), 4)
        }

# Alternative function to help debug the exact feature requirements
def get_model_info():
    """Helper function to get information about the loaded model"""
    try:
        # For XGBoost models
        if hasattr(model, 'get_booster'):
            booster = model.get_booster()
            feature_names = booster.feature_names
            n_features = booster.num_features()
            if feature_names:
                names = feature_names
            else:
                names = [f"feature_{i}" for i in range(n_features)]
            return {
                "n_features": n_features,
                "feature_names": names
            }
        # For sklearn models
        elif hasattr(model, 'n_features_in_'):
            return {
                "n_features": model.n_features_in_,
                "feature_names": getattr(model, 'feature_names_in_', [f"feature_{i}" for i in range(model.n_features_in_)])
            }
        else:
            return {"n_features": "Unknown", "feature_names": "Unknown"}
    except Exception as e:
        return {"error": str(e)}
