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

## UI

The application provides an interactive web interface for blood donation predictions. Below are screenshots showcasing the prediction interface:

### Prediction Example 1: "Will Not Donate"

![Prediction Example 1](https://github.com/Ratnesh-3/Blood-Donation-Prediction/raw/main/ui_screenshot_1.png)

*Input: Months since last donation: 12, Number of donations: 23, Total volume donated: 12500 cc, Months since first donation: 9*
*Result: Will Not Donate*

### Prediction Example 2: "Will Donate"

![Prediction Example 2](https://github.com/Ratnesh-3/Blood-Donation-Prediction/raw/main/ui_screenshot_2.png)

*Input: Months since last donation: 2, Number of donations: 20, Total volume donated: 5000 cc, Months since first donation: 45*
*Result: Will Donate*

### Prediction Example 3: "Will Donate" (High Frequency Donor)

![Prediction Example 3](https://github.com/Ratnesh-3/Blood-Donation-Prediction/raw/main/ui_screenshot_3.png)

*Input: Months since last donation: 2, Number of donations: 50, Total volume donated: 12500 cc, Months since first donation: 98*
*Result: Will Donate*

## Screenshots

Below are example screenshots you can add to showcase the app UI and visualizations. The image files are referenced from the `screenshots/` directory — add your PNGs there with the filenames use[...]

### Prediction API (example request/response)

![API prediction example](screenshots/api_prediction.png)

*Caption: Example curl request and the JSON response returned by the Flask `/predict` endpoint.*

### Gradio Demo / Input Form

![Gradio input form](screenshots/gradio_input.png)

*Caption: Example of the demo input form (from workbook.py) where the user enters features to get a prediction.*

### Visualizations

![Feature distribution and correlations](screenshots/visualizations.png)

*Caption: Example exploratory plots (boxplots, histograms, correlation heatmap) used during EDA and feature engineering.*

<img width="950" height="762" alt="WhatsApp Image 2026-07-31 at 11 32 14 AM" src="https://github.com/user-attachments/assets/a752fbdb-e75f-4821-ba6e-1f7804fbe9ab" />
<img width="950" height="762" alt="WhatsApp Image 2026-07-31 at 11 32 14 AM" src="https://github.com/user-attachments/assets/5e249cee-7810-4737-bf94-908ff6ab382b" />
<img width="950" height="762" alt="WhatsApp Image 2026-07-31 at 11 32 14 AM" src="https://github.com/user-attachments/assets/9a76c4e2-75c0-4f7d-bf56-14a3d0d435f7" />



How to add your screenshots

1. Create a `screenshots/` directory at the project root.
2. Save your images with these filenames (or update the README to point to your filenames):
   - `screenshots/api_prediction.png`
   - `screenshots/gradio_input.png`
   - `screenshots/visualizations.png`
3. Commit and push the images to the repository. Example git commands:

```bash
mkdir screenshots
# copy your images into screenshots/
git add screenshots/api_prediction.png screenshots/gradio_input.png screenshots/visualizations.png
git commit -m "Add README screenshots"
git push origin main
```

If you'd like, I can add the screenshots for you — upload the image files here (or give me public URLs) and tell me the branch name to use; I'll create a branch, add the images under `screensho[...]
