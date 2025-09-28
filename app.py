import os, pickle
from flask import Flask, request, jsonify, render_template
import pandas as pd

BASE = os.path.dirname(__file__)
ART = os.path.join(BASE, "artifacts")

def load_pickle(name):
    path = os.path.join(ART, name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing pickle: {path}")
    return pickle.load(open(path, "rb"))

# Load models & data
TFIDF = load_pickle("tfidf_vectorizer.pkl")
RF_MODEL = load_pickle("rf_model.pkl")
USER_ITEM = load_pickle("user_item_matrix.pkl")
ITEM_SIM = load_pickle("item_similarity.pkl")
META = load_pickle("product_metadata.pkl") if os.path.exists(os.path.join(ART, "product_metadata.pkl")) else {}
USER_META = load_pickle("user_metadata.pkl") if os.path.exists(os.path.join(ART, "user_metadata.pkl")) else {}

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/predict", methods=["POST"])
def predict():
    data = request.json or request.form
    text = (data.get("text") or "").strip()
    user_name = (data.get("user") or "").strip().lower()
    item_name = (data.get("item") or "").strip().lower()

    # --- Sentiment ---
    sentiment = {"label": "neutral", "prob_positive": None}
    if text:
        vec = TFIDF.transform([text])
        pred = RF_MODEL.predict(vec)[0]
        prob = float(RF_MODEL.predict_proba(vec)[0][1]) if hasattr(RF_MODEL, "predict_proba") else None
        sentiment = {"label": "positive" if pred == 1 else "negative", "prob_positive": prob}

    # --- Dynamic Recommendations ---
    recs_dict = {}

    # User-based recommendations
    user_id = next((uid for uid, info in USER_META.items() if info.get("name", "").lower() == user_name), None)
    if user_id and user_id in USER_ITEM.index:
        user_row = USER_ITEM.loc[user_id]
        liked = user_row[user_row >= 4].index.tolist()
        if liked:
            scores = pd.Series(0.0, index=ITEM_SIM.columns)
            for it in liked:
                scores = scores.add(ITEM_SIM[it] * user_row[it], fill_value=0)
            scores[list(user_row[user_row > 0].index)] = -1
            scores = scores.sort_values(ascending=False).head(10)
            for idx in scores.index:
                recs_dict[idx] = {"score": float(scores.loc[idx]), "title": META.get(idx, {}).get("name", str(idx))}

    # Item-based recommendations
    item_id = next((iid for iid, info in META.items() if info.get("name", "").lower() == item_name), None)
    if item_id and item_id in ITEM_SIM.columns:
        scores = ITEM_SIM[item_id].sort_values(ascending=False).head(6)
        for idx in scores.index:
            if idx != item_id:
                if idx in recs_dict:
                    recs_dict[idx]["score"] += float(scores.loc[idx])
                else:
                    recs_dict[idx] = {"score": float(scores.loc[idx]), "title": META.get(idx, {}).get("name", str(idx))}

    # Fallback popular items
    if not recs_dict:
        avg = USER_ITEM.replace(0, float("nan")).mean().sort_values(ascending=False).head(5)
        for i in avg.index:
            recs_dict[i] = {"score": float(avg.loc[i]), "title": META.get(i, {}).get("name", str(i))}

    recs = sorted(recs_dict.values(), key=lambda x: x["score"], reverse=True)[:5]

    return jsonify({"sentiment": sentiment, "recommendations": recs, "mode": "dynamic"})

if __name__ == "__main__": app.run(debug=True, host="0.0.0.0", port=5000)
