from flask import Flask, render_template, request
import pickle
import numpy as np

# Load model, scaler, and PCA
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
pca = pickle.load(open("pca.pkl", "rb"))

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    if request.method == "POST":
        try:
            # Get input values
            features = [
                float(request.form["tempo"]),
                float(request.form["loudness"]),
                float(request.form["danceability"]),
                float(request.form["energy"]),
                float(request.form["valence"]),
                float(request.form["acousticness"]),
                float(request.form["instrumentalness"]),
                float(request.form["speechiness"]),
                float(request.form["liveness"]),
                float(request.form["duration_ms"])
            ]

            # Scale → PCA → Predict
            scaled = scaler.transform([features])
            pca_features = pca.transform(scaled)
            prediction = model.predict(pca_features)[0]
        except:
            prediction = "Invalid input! Please enter numbers only."
    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
