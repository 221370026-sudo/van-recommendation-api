from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import pandas as pd
import os

app = Flask(__name__)
CORS(app)

# Load model
model = pickle.load(open("van_model.pkl", "rb"))

@app.route("/")
def home():
    return "Van Recommendation API is running"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    df = pd.DataFrame(data)

    scores = model.predict(df)

    return jsonify({"scores": scores.tolist()})

# IMPORTANT for Railway
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
