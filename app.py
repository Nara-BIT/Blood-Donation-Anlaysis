import numpy as np
import joblib
from flask import Flask, request, jsonify
from flask_cors import CORS

print("Loading model and scaler...")
# Load the pre-trained model and scaler
try:
    model = joblib.load('blood_donation_xgb_model.joblib')
    scaler = joblib.load('blood_donation_scaler.joblib')
except FileNotFoundError:
    print("Error: Model or scaler files not found.")
    print("Please run 'train.py' first to create the model files.")
    exit()

print("Model and scaler loaded successfully.")

# Initialize the Flask application
app = Flask(__name__)
# Enable CORS (Cross-Origin Resource Sharing)
CORS(app)

# This is the prediction function, adapted from your code
def predict_donation(months_last, num_donations, total_volume, months_first):
    
    # 1. Engineer features from the inputs
    # Add 1 to avoid division by zero
    donation_rate = num_donations / (months_first + 1)
    recency_inverse = 1 / (months_last + 1)

    # 2. Create the feature array in the correct order
    # This MUST match the order from the training script
    features = np.array([months_last, num_donations, total_volume, months_first,
                         donation_rate, recency_inverse]).reshape(1, -1)

    # 3. Scale the features using the loaded scaler
    features_scaled = scaler.transform(features)
    
    # 4. Make the prediction using the loaded model
    pred = model.predict(features_scaled)
    proba = model.predict_proba(features_scaled)

    # 5. Return the result
    if pred[0] == 1:
        confidence = proba[0][1]
        result_text = "Will Donate"
    else:
        confidence = proba[0][0]
        result_text = "Will Not Donate"
        
    return result_text, confidence

# Define an API endpoint for prediction
@app.route('/predict', methods=['POST'])
def handle_prediction():
    try:
        # Get the JSON data sent from the frontend
        data = request.json
        
        # Extract features
        features_list = [
            data['months_last'],
            data['num_donations'],
            data['total_volume'],
            data['months_first']
        ]

        # Get the prediction
        result_text, confidence = predict_donation(*features_list)

        # Send the response back as JSON
        return jsonify({
            'prediction_text': result_text,
            'confidence': confidence
        })

    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({'error': 'An error occurred during prediction.'}), 500

# Run the Flask server
if __name__ == '__main__':
    print("Starting Flask server at http://127.0.0.1:5000")
    app.run(debug=True, port=5000)
