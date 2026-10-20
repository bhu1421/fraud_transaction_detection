import streamlit as st
from src.fraud_detection.inference import predict_fraud

st.set_page_config(page_title="Fraud Transaction Detector", layout="centered")

st.title("🔍 FBI Fraud Transaction Detection App")
st.markdown("Enter the transaction details below to check if it's fraudulent.")

# Input fields
step = st.number_input("Transaction Step (Time)", min_value=0, value=1)
amount = st.number_input("Transaction Amount", min_value=0.0)
oldbalanceOrg = st.number_input("Old Balance (Origin)", min_value=0.0)
newbalanceOrig = st.number_input("New Balance (Origin)", min_value=0.0)
oldbalanceDest = st.number_input("Old Balance (Destination)", min_value=0.0)
newbalanceDest = st.number_input("New Balance (Destination)", min_value=0.0)

# Transaction type options (based on your dataset)
type_options = ["CASH_OUT", "TRANSFER", "PAYMENT", "DEBIT", "CASH_IN"]
type_option = st.selectbox("Transaction Type", type_options)

# Additional engineered features that your model expects
st.subheader("Additional Transaction Details")
step_hour = st.number_input("Step Hour (Hour of day when transaction occurred)", min_value=0, max_value=23, value=1)
isMerchant = st.checkbox("Is Merchant Account", value=False)
isEmptied = st.checkbox("Account Emptied After Transaction", value=True)  # Default True based on your data
emptyReceiver = st.checkbox("Receiver Account Empty", value=False)

# Predict Button
if st.button("Predict Fraud"):
    # Encode transaction type as numeric (based on your preprocessing)
    type_mapping = {"TRANSFER": 1, "CASH_OUT": 1, "PAYMENT": 1, "DEBIT": 1, "CASH_IN": 1}
    type_encoded = type_mapping.get(type_option, 1)
    
    # Create one-hot encoding for specific types (only the ones your model uses)
    type_CASH_OUT = 1 if type_option == "CASH_OUT" else 0
    type_TRANSFER = 1 if type_option == "TRANSFER" else 0
    
    # Create input data matching your model's exact feature set
    input_data = {
        "step": step,
        "type": type_encoded,
        "amount": amount,
        "oldbalanceOrg": oldbalanceOrg,
        "newbalanceOrig": newbalanceOrig,
        "oldbalanceDest": oldbalanceDest,
        "newbalanceDest": newbalanceDest,
        "step_hour": step_hour,
        "isMerchant": int(isMerchant),
        "isEmptied": int(isEmptied),
        "emptyReceiver": int(emptyReceiver),
        "type_CASH_OUT": type_CASH_OUT,
        "type_TRANSFER": type_TRANSFER
    }
    
    try:
        prediction = predict_fraud(input_data)
        
        # Display results with better formatting
        if prediction["prediction"] == 1:
            st.error(f"🚨 **FRAUDULENT TRANSACTION DETECTED!**")
            st.error(f"Fraud Probability: **{prediction['probability_of_fraud']:.2%}**")
        else:
            st.success(f"✅ **LEGITIMATE TRANSACTION**")
            st.success(f"Fraud Probability: **{prediction['probability_of_fraud']:.2%}**")
            
        # Display input summary
        st.write("### Transaction Summary")
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**Amount:** ${amount:,.2f}")
            st.write(f"**Type:** {type_option}")
            st.write(f"**Origin Balance:** ${oldbalanceOrg:,.2f} → ${newbalanceOrig:,.2f}")
            st.write(f"**Merchant:** {'Yes' if isMerchant else 'No'}")
        with col2:
            st.write(f"**Step:** {step} (Hour: {step_hour})")
            st.write(f"**Account Emptied:** {'Yes' if isEmptied else 'No'}")
            st.write(f"**Dest Balance:** ${oldbalanceDest:,.2f} → ${newbalanceDest:,.2f}")
            st.write(f"**Empty Receiver:** {'Yes' if emptyReceiver else 'No'}")
            
    except Exception as e:
        st.error(f"An error occurred during prediction: {str(e)}")
        
        # Debug information
        st.write("### Debug Info")
        st.write(f"Number of features provided: {len(input_data)}")
        st.write("Features:", list(input_data.keys()))
        st.write("Feature values:", list(input_data.values()))
