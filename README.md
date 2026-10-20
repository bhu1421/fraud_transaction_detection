# Financial Fraud Detection in  Transactions

A machine learning prototype that analyzes transaction data, compares fraud classifiers, and includes a Streamlit interface for entering transaction details.

> **Status:** The notebook contains a saved experimental evaluation. The Streamlit inference path still needs its model artifact and input features aligned with the training pipeline before its predictions can be relied on.

## Project overview

The analysis explores fraud patterns, creates balance and transaction features, and evaluates Logistic Regression, Random Forest, and XGBoost. It also visualizes XGBoost explanations with SHAP.

## Dataset

The notebook reports 6,362,620 transactions and 8,213 fraud cases, approximately 0.129% of the data. It reports no missing values. The transaction types are CASH_IN, CASH_OUT, DEBIT, PAYMENT, and TRANSFER.

The dataset file is expected at data/raw/Fraud.csv. The notebook should be run with the repository root as its working directory if its data-loading cell uses that path.

## Findings from the saved notebook run

Fraud cases in the saved analysis occurred in CASH_OUT and TRANSFER transactions:

| Type | Fraud cases | Fraud rate within type |
| --- | ---: | ---: |
| TRANSFER | 4,097 | 0.769% |
| CASH_OUT | 4,116 | 0.184% |
| CASH_IN, DEBIT, PAYMENT | 0 | 0% in this dataset |

The fraud counts are nearly evenly split between TRANSFER and CASH_OUT. These observed results do not establish that other transaction types can never be fraudulent.

The saved analysis reports a median fraud transaction amount of about 441,423 dataset units, a mean of about 1,467,967, a 90th percentile of about 4,521,724, and a maximum of 10,000,000.

The notebook's isEmptied feature is true for about 97.55% of fraud cases and 23.80% of legitimate cases. Its emptyReceiver feature is true for about 15.52% of fraud cases and 6.07% of legitimate cases. These percentages correspond to the feature definitions in the notebook; they should not be conflated with other zero-balance definitions.

Only 16 fraud cases had isFlaggedFraud set in the saved analysis. The modeling features exclude that field.

## Feature engineering and training

The notebook derives features including :

- step_hour: step modulo 24.
- isMerchant: whether the destination identifier starts with M.
- isEmptied: the origin account had a positive opening balance and ends at zero.
- emptyReceiver: the destination opening balance is zero and its ending balance is positive.
- errorBalanceOrig: newbalanceOrig + amount - oldbalanceOrg.
- errorBalanceDest: oldbalanceDest + amount - newbalanceDest.

The final modeling data filters to TRANSFER and CASH_OUT. It uses an 80/20 stratified random train/test split with random_state=42 and applies SMOTE to the training split. The saved test split has 554,082 transactions, including 1,643 fraud cases.

The project also used Logistic Regression. However, the current saved notebook does not contain its training cell or saved evaluation metrics. The quantitative comparison below therefore includes Random Forest and XGBoost, whose results are present in the notebook. The notebook does not show a completed hyperparameter search. The final model matrix contains 13 columns, including one-hot encoded type_TRANSFER and the two balance-error features. Use the notebook itself as the source of truth for the exact feature construction.

## Model results

These figures are from the notebook's saved holdout outputs:

| Model | True positives | False positives | False negatives | ROC AUC | PR AUC |
| --- | ---: | ---: | ---: | ---: | ---: |
| Random Forest | 1,638 | 11 | 5 | 0.99907 | 0.99708 |
| XGBoost | 1,637 | 29 | 6 | 0.99899 | 0.99793 |
| Logistic Regression | 1,555 | 20,678 | 88 | 0.99031 | 0.67058 |

Random Forest made fewer false-positive and false-negative predictions at the notebook's default classification threshold. XGBoost had a slightly higher PR AUC. Neither result proves how the models will perform on future transactions: the notebook uses a random holdout from the same dataset, not a later time period or an independent dataset.

Accuracy alone is not a useful model-selection measure here because fraud is rare. Compare precision, recall, PR AUC, and the number of transactions sent for review at the chosen threshold.

## Model explanations

The notebook includes SHAP summary plots for XGBoost. The plots can help explore which features influence predictions. The README does not assign numerical feature ranks or claim particular risk thresholds because those claims are not established by the saved notebook outputs.

## Streamlit app and Docker

The Streamlit app is in app/streamlit_app.py. From the repository root, install dependencies and launch it with:

    python -m pip install -r requirements.txt
    python -m streamlit run app/streamlit_app.py

The inference module expects the model at artifacts/xgb_fraud_detection_model.pkl. A model file is currently located under notebooks/artifacts/, which is a different path. The Dockerfile also expects the root artifacts path.

The notebook's trained feature matrix and the feature list assembled by the inference module do not currently match. Align the model artifact path and feature preprocessing before using app predictions. The Docker image is not buildable until the expected root artifact is supplied.

To build and run the Docker image after resolving those requirements:

    docker build -t fraud-transaction-detection .
    docker run --rm -p 8501:8501 fraud-transaction-detection

Then open http://localhost:8501.

## Repository layout

- app/ - Streamlit interface.
- data/ - dataset notes and raw data location.
- notebooks/ - analysis, training, evaluation, and saved notebook artifact.
- src/fraud_detection/ - inference code.
- tests/ - smoke test.

## Limitations and next steps

- Evaluate with a time-based split and, if possible, an independent dataset.
- Select a probability threshold based on the cost of missed fraud and false alarms.
- Align training and inference feature names, order, and transformations.
- Confirm the model file and dependency versions before deployment.
- Monitor performance over time and retrain only after evaluation.

The results above describe one saved notebook run; they are not a production guarantee or measured financial-loss prevention.
