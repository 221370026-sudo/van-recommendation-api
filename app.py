@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        # convert safely
        df = pd.DataFrame(data)

        # force correct column order (IMPORTANT)
        expected_cols = [
            "distance_to_user_km",
            "eta_min",
            "seats_available",
            "route_match_score"
        ]

        df = df[expected_cols]

        # ensure numeric types
        df = df.astype(float)

        preds = model.predict(df)

        return jsonify({
            "scores": preds.tolist()
        })

    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 500
