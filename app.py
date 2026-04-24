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
# PREDICT ROUTE (FIXED)
# ---------------------------
@app.route("/predict", methods=["POST"])
def predict():
    try:
        # get JSON input
        data = request.get_json(force=True)

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

        # enforce correct order
        df = df[required_cols]

        # ensure numeric format
        df = df.astype(float)

        # prediction using probability (BEST for ranking)
        scores = model.predict_proba(df)[:, 1]

        return jsonify({
            "scores": scores.tolist()
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
    
