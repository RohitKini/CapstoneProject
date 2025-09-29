# train_n_pickle.py
import os, pickle
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity

DATA_CSV = "sample30.csv"   # your dataset
OUT_DIR = "artifacts"
os.makedirs(OUT_DIR, exist_ok=True)

# -----------------------------
# Load Data
# -----------------------------
df = pd.read_csv(DATA_CSV)   
# expected cols: reviews_text, user_sentiment, reviews_username, id, reviews_rating, name

# Basic cleaning
df['reviews_text'] = df['reviews_text'].fillna("").astype(str)
df['reviews_title'] = df['reviews_title'].fillna("").astype(str)
df['reviews_username'] = df['reviews_username'].fillna("unknown").astype(str)
df['name'] = df['name'].fillna("unknown").astype(str)

# Create binary labels: positive -> 1, negative -> 0
if 'user_sentiment' in df.columns:
    df['label'] = df['user_sentiment'].map({'positive': 1, 'negative': 0}).fillna(0).astype(int)
else:
    df['label'] = (df['reviews_rating'] >= 3).astype(int)

# -----------------------------
# Train TF-IDF + RandomForest
# -----------------------------
tfidf = TfidfVectorizer(max_features=20000, ngram_range=(1,2), min_df=3)
X = tfidf.fit_transform(df['reviews_title'] + " " + df['reviews_username'] + " " + df['name'])
y = df['label'].values

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.15, random_state=42, stratify=y
)

rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)

# Save models
pickle.dump(tfidf, open(os.path.join(OUT_DIR, "tfidf_vectorizer.pkl"), "wb"))
pickle.dump(rf, open(os.path.join(OUT_DIR, "rf_model.pkl"), "wb"))

# -----------------------------
# User-Item Matrix
# -----------------------------
ui = df.pivot_table(
    index='reviews_username',
    columns='id',
    values='reviews_rating',
    aggfunc='mean'
).fillna(0)

pickle.dump(ui, open(os.path.join(OUT_DIR, "user_item_matrix.pkl"), "wb"))

# -----------------------------
# Item similarity
# -----------------------------
item_sim = cosine_similarity(ui.T)
item_sim_df = pd.DataFrame(item_sim, index=ui.columns, columns=ui.columns)
pickle.dump(item_sim_df, open(os.path.join(OUT_DIR, "item_similarity.pkl"), "wb"))

# -----------------------------
# Product metadata (id -> name)
# -----------------------------
if 'name' in df.columns:
    meta = df.drop_duplicates(subset='id').set_index('id')[['name']].to_dict(orient='index')
else:
    meta = {}
pickle.dump(meta, open(os.path.join(OUT_DIR, "product_metadata.pkl"), "wb"))

# -----------------------------
# User metadata (username -> stats)
# -----------------------------
user_meta = (
    df.groupby('reviews_username')
      .agg(
          username=('reviews_username', 'first'),        
          total_reviews=('id', 'count'),
          avg_rating=('reviews_rating', 'mean'),
          positive_reviews=('label', 'sum'),
          negative_reviews=('label', lambda x: (x == 0).sum())
      )
      .reset_index()
)

user_meta_dict = user_meta.set_index('reviews_username').to_dict(orient='index')
pickle.dump(user_meta_dict, open(os.path.join(OUT_DIR, "user_metadata.pkl"), "wb"))

# -----------------------------
print("✅ Pickles saved to", OUT_DIR)
print("Files:", os.listdir(OUT_DIR))
