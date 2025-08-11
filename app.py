from flask import Flask, render_template, request
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import numpy as np
import joblib

app = Flask(__name__)

# Load dataset
df = pd.read_csv('pca.csv')

# Features
X = df.values

# Scale data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# PCA reduce to 2 components
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Cluster AQI into 3 categories: Good, Moderate, Poor
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(X_pca)

# Save models
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(pca, 'pca.pkl')
joblib.dump(kmeans, 'kmeans.pkl')

# AQI category mapping (based on cluster label)
category_map = {0: "Good", 1: "Moderate", 2: "Poor"}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get user inputs
        pm25 = float(request.form['pm25'])
        pm10 = float(request.form['pm10'])
        no2 = float(request.form['no2'])
        so2 = float(request.form['so2'])
        co = float(request.form['co'])
        o3 = float(request.form['o3'])

        scaler = joblib.load('scaler.pkl')
        pca = joblib.load('pca.pkl')
        kmeans = joblib.load('kmeans.pkl')

        features = np.array([[pm25, pm10, no2, so2, co, o3]])
        features_scaled = scaler.transform(features)
        features_pca = pca.transform(features_scaled)
        cluster = kmeans.predict(features_pca)[0]

        category = category_map.get(cluster, "Unknown")

        return render_template('index.html', prediction_text=f"AQI Category: {category}")
    except Exception as e:
        return str(e)

if __name__ == '__main__':
    app.run(debug=True)
