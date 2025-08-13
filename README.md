-----

#  Music Genre Classification using PCA & KNN

##  Project Overview

This project uses **Principal Component Analysis (PCA)** for dimensionality reduction and **K-Nearest Neighbors (KNN)** for classification to predict the genre of music based on audio features.

It is deployed as a **Flask web application** with a simple HTML & CSS interface.

-----

##  Technologies Used

  - Python 3.x
  - **Flask** – Web framework for deployment
  - **Scikit-learn** – Machine learning library (PCA, KNN)
  - **Pandas** – Data manipulation
  - **NumPy** – Numerical computation
  - **HTML & CSS** – Frontend interface

-----

##  Project Structure

```
music_genre_pca_knn/
│── model.py           # Machine learning model training (PCA + KNN)
│── app.py             # Flask application
│── templates/
│   └── index.html     # Frontend HTML form
│── static/
│   └── style.css      # CSS styling
│── music_genre_dataset.csv   # Dataset with 200 rows
│── README.md          # Project documentation
```

-----

##  Dataset Description

The dataset contains 200 rows of synthetic music features and labels.

**Columns:**

  - `tempo` – Beats per minute (BPM) of the track
  - `energy` – Energy level of the track (0–1)
  - `danceability` – Danceability score (0–1)
  - `acousticness` – Acousticness score (0–1)
  - `instrumentalness` – Instrumental score (0–1)
  - `liveness` – Liveness score (0–1)
  - `valence` – Positivity of the track (0–1)
  - `genre` – Target label (Rock, Jazz, Classical, Hip-Hop, Pop)

-----

##  Installation & Setup

1.  **Clone the repository**

    ```bash
    git clone https://github.com/yourusername/music-genre-pca-knn.git
    cd music-genre-pca-knn
    ```

2.  **Install dependencies**

    ```bash
    pip install flask pandas numpy scikit-learn
    ```

3.  **Run model training**

    ```bash
    python model.py
    ```

    This will train the PCA + KNN model and save it as `pca_knn_model.pkl`.

4.  **Run Flask app**

    ```bash
    python app.py
    ```

    Flask will start running on: `http://127.0.0.1:5000/`

-----

##  Usage

1.  Open the Flask app in your browser.
2.  Enter music features (tempo, energy, danceability, etc.).
3.  Click **Predict Genre**.
4.  The model will output the predicted music genre.

-----

##  Model Details

  - **PCA:** Reduces dimensionality of features for faster computation.
  - **KNN:** Classifies music based on feature similarity.
  - `n_neighbors`: 5 (can be tuned for better performance).

-----

##  Screenshot
---
  Home page :
    <img width="549" height="498" alt="Screenshot 2025-08-13 101016" src="https://github.com/user-attachments/assets/87ced038-62b5-4357-8b38-766460640bba" />
---
  Predection page
    <img width="608" height="551" alt="Screenshot 2025-08-13 101026" src="https://github.com/user-attachments/assets/f138991d-4621-45a2-ad21-d6bd531ebf05" />
