from flask import Flask, request, jsonify, send_from_directory
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__, static_folder="static")

# ==========================================
# LOAD DATA & MODELS ONCE (Critical for memory)
# ==========================================

# Load only needed columns
books_df = pd.read_csv("books.csv", usecols=[0, 1, 2, 3])

# Create combined text feature
books_df["features"] = (
    books_df["title"].astype(str) + " " + books_df["authors"].astype(str)
)

# Initialize TF-IDF model
tfidf = TfidfVectorizer(stop_words="english")
tfidf_matrix = tfidf.fit_transform(books_df["features"])

# ==========================================
# Recommendation Logic
# ==========================================

def weighted_recommend(book_title, weight, n=50):
    """Compute similarity only for the selected book (MUCH lower RAM)."""
    try:
        index = books_df[books_df["title"] == book_title].index[0]
    except IndexError:
        return []

    # compute similarity only against ONE row instead of N×N matrix
    sims = cosine_similarity(tfidf_matrix[index], tfidf_matrix)[0]

    # apply weight
    sims = sims * weight

    # get top N similar indices (excluding itself)
    top_indices = sims.argsort()[::-1][1:n+1]

    # return titles
    return books_df.iloc[top_indices]["title"].tolist()


# ==========================================
# ROUTES
# ==========================================

@app.route("/")
def serve_frontend():
    return send_from_directory("static", "index.html")

@app.route("/recommend", methods=["POST"])
def recommend():
    data = request.json
    user_prefs = data["preferences"]  # [{title, rating}, ...]

    final_scores = {}
    user_titles = [p["title"] for p in user_prefs]

    for pref in user_prefs:
        title = pref["title"]
        rating = pref["rating"]

        recs = weighted_recommend(title, rating, n=50)

        for r in recs:
            if r not in user_titles:
                final_scores[r] = final_scores.get(r, 0) + rating

    ranked = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
    ranked = [r[0] for r in ranked[:10]]  # top 10

    return jsonify({"recommendations": ranked})

# ==========================================
# RUN LOCAL ONLY (Render uses gunicorn)
# ==========================================

import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)



       
    


