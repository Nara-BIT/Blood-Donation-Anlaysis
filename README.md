# Blood Donation Analysis

This project predicts whether a person will donate blood based on donation history features. The repository includes:

- `train.py` to train the model and save the artifacts
- `app.py` to load the saved model and expose a Flask prediction API

## Setup

Install the required packages before running any script:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn xgboost scikit-optimize flask flask-cors joblib
```

## How to Run

### 1. Train the model

Run the training script from the project root:

```bash
python train.py
```

This will:

- load `blood-train.csv`
- perform preprocessing, feature engineering, SMOTE balancing, and XGBoost tuning
- save the trained model to `blood_donation_xgb_model.joblib`
- save the scaler to `blood_donation_scaler.joblib`

### 2. Start the prediction API

After training completes, run:

```bash
python app.py
```

The Flask server starts on `http://127.0.0.1:5000`.

### 3. Call the prediction endpoint

Send a `POST` request to `/predict` with JSON input:

```bash
curl -X POST http://127.0.0.1:5000/predict ^
  -H "Content-Type: application/json" ^
  -d "{\"months_last\":10,\"num_donations\":5,\"total_volume\":1250,\"months_first\":36}"
```

Example response:

```json
{
  "prediction_text": "Will Donate",
  "confidence": 0.85
}
```

## Model Performance

| Model | Accuracy | Precision | Recall | F1-Score |
| --- | ---: | ---: | ---: | ---: |
| Logistic Regression | 0.7625 | 0.8234 | 0.5463 | 0.6667 |
| Decision Tree | 0.7891 | 0.8148 | 0.6082 | 0.6982 |
| Random Forest | 0.8061 | 0.8378 | 0.6642 | 0.7417 |
| Support Vector Classifier | 0.7823 | 0.8095 | 0.5864 | 0.6789 |
| XGBoost (Optimized) | 0.8300 | 0.8547 | 0.8696 | 0.8620 |

## Best Model

`XGBoost (Optimized)` performed best across all four metrics and is the model saved by the training pipeline.

## Notes

- Run the scripts from the project root so the CSV and model paths resolve correctly.
- If you retrain the model, the `.joblib` files will be overwritten with the new artifacts.

## UI Screenshots

The application provides an interactive web interface for blood donation predictions. Below are screenshots showcasing the prediction interface:

### Prediction Example 1: "Will Not Donate"

<img width="925" height="755" alt="WhatsApp Image 2026-07-31 at 11 32 20 AM" src="https://github.com/user-attachments/assets/511359dd-4124-409e-929a-619f8ab78c9d" />


*Input: Months since last donation: 12, Number of donations: 23, Total volume donated: 12500 cc, Months since first donation: 9*
*Result: Will Not Donate*

### Prediction Example 2: "Will Donate"

<img width="950" height="762" alt="WhatsApp Image 2026-07-31 at 11 32 14 AM" src="https://github.com/user-attachments/assets/b33d33ce-ccf7-4d3e-b8a9-de625cf89b20" />


*Input: Months since last donation: 2, Number of donations: 20, Total volume donated: 5000 cc, Months since first donation: 45*
*Result: Will Donate*

### Prediction Example 3: "Will Donate" (High Frequency Donor)

<img width="1471" height="893" alt="WhatsApp Image 2026-07-31 at 11 32 06 AM" src="https://github.com/user-attachments/assets/914b30c7-86e3-497f-bc8f-2393b08f36e5" />

*Input: Months since last donation: 2, Number of donations: 50, Total volume donated: 12500 cc, Months since first donation: 98*
*Result: Will Donate*

