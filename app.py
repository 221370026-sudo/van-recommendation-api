from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import pandas as pd
import os

# IMPORTANT: CREATE APP FIRST
app = Flask(__name__)
CORS(app)

# LOAD MODEL
model = pickle.load(open("van_model.pkl", "rb"))

# HOME ROUTE
@app.route("/")
def home():
    return "Van Recommendation API is running"

# PREDICT ROUTE
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        df = pd.DataFrame(data)

        expected_cols = [
            "distance_to_user_km",
            "eta_min",
            "seats_available",
            "route_match_score"
        ]

        df = df[expected_cols]
        df = df.astype(float)

        preds = model.predict(df)

        return jsonify({"scores": preds.tolist()})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# RAILWAY ENTRY POINT
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
