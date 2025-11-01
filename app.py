from flask import Flask, request, jsonify
import joblib
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load the saved model
model_path = "artifacts/best_model.pkl"

model = joblib.load(model_path)

# Define home route
@app.route('/')
def home():
    return "🏠 Welcome to the California Housing Price Prediction API!"

# Define a prediction route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data from request
        data = request.get_json()
        # Example input: {"features": [8.3252, 41.0, 6.984127, 1.02381, 322.0, 2.555556, 37.88, -122.23]}
        features = np.array(data['features']).reshape(1, -1)

        # Make prediction
        prediction = model.predict(features)[0]

        # Return the result
        return jsonify({
            "predicted_median_house_value": round(prediction, 2)
        })
    
    except Exception as e:
        return jsonify({"error": str(e)})

# Run the app
if __name__ == "__main__":
    app.run(debug=True)
