from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import pandas as pd
import os

# ---------------------------
# INIT APP
# ---------------------------
app = Flask(__name__)
CORS(app)

# ---------------------------
# LOAD MODEL
# ---------------------------
model = pickle.load(open("van_model.pkl", "rb"))

# ---------------------------
# HOME ROUTE
# ---------------------------
@app.route("/")
def home():
    return "Van Recommendation API is running"

# ---------------------------
# PREDICT ROUTE
# ---------------------------
@app.route("/predict", methods=["POST"])
def predict():
    try:
        # get JSON input (must be list of dicts)
        data = request.get_json(force=True)

        if not isinstance(data, list) or len(data) == 0:
            return jsonify({"error": "Input must be a non-empty JSON array"}), 400

        # convert to DataFrame
        df = pd.DataFrame(data)

        # REQUIRED COLUMNS (must match training exactly)
        required_cols = [
            "user_lat",
            "user_lng",
            "dest_lat",
            "dest_lng",
            "van_lat",
            "van_lng",
            "distance_to_user_km",
            "eta_min",
            "seats_available",
            "route_match_score"
        ]

        # ensure all columns are present
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            return jsonify({"error": f"Missing required fields: {missing}"}), 400

        # enforce correct order + numeric format
        df = df[required_cols].astype(float)

        # XGBRanker => predict() (not predict_proba)
        # If model supports predict_proba, use probability score
        if hasattr(model, "predict_proba"):
            scores = model.predict_proba(df)[:, 1]
        else:
            scores = model.predict(df)

        return jsonify({
            "scores": [float(s) for s in scores]
        })

    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 500

# ---------------------------
# RUN SERVER (RAILWAY)
# ---------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
