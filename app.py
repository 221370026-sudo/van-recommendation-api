from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import pandas as pd

app = Flask(__name__)
CORS(app)

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

# IMPORTANT: no app.run() on Render