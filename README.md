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

## Screenshots

Below are example screenshots you can add to showcase the app UI and visualizations. The image files are referenced from the `screenshots/` directory — add your PNGs there with the filenames used below, or update the paths to match your files.

### Prediction API (example request/response)

![API prediction example](screenshots/api_prediction.png)

*Caption: Example curl request and the JSON response returned by the Flask `/predict` endpoint.*

### Gradio Demo / Input Form

![Gradio input form](screenshots/gradio_input.png)

*Caption: Example of the demo input form (from workbook.py) where the user enters features to get a prediction.*

### Visualizations

![Feature distribution and correlations](screenshots/visualizations.png)

*Caption: Example exploratory plots (boxplots, histograms, correlation heatmap) used during EDA and feature engineering.*


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

If you'd like, I can add the screenshots for you — upload the image files here (or give me public URLs) and tell me the branch name to use; I'll create a branch, add the images under `screenshots/`, and update the README to point to them in a single commit.
