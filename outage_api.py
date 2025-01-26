from flask import Flask, request, jsonify
from catboost import CatBoostClassifier
import pandas as pd

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
model = CatBoostClassifier()
model.load_model("catboost_model_improved.bin")

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    df = pd.DataFrame([data])  # Convert input data to DataFrame
    prediction = model.predict(df)
    return jsonify({'outage': int(prediction[0])})

if __name__ == '__main__':
    app.run(debug=True)
