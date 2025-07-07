from flask import Flask, request, render_template
import joblib
import numpy as np
import os

app = Flask(__name__)

# Load model
model = joblib.load('heart_disease_model.pkl')

@app.route('/')
def home():
    # Pass prediction_text as None to the template
    return render_template('index.html', prediction_text=None)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Collect form inputs
        features = [float(x) for x in request.form.values()]
        prediction = model.predict([np.array(features)])
        output = 'Heart Disease Detected' if prediction[0] == 1 else 'No Heart Disease'
    except Exception as e:
        output = f"Error: {str(e)}"

    # Return the prediction text to the template
    return render_template('index.html', prediction_text=output)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # For Render/Heroku compatibility
    app.run(host='0.0.0.0', port=port)