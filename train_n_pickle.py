# train_n_pickle.py
import os, pickle
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity

DATA_CSV = "sample30.csv"   # put your file here
OUT_DIR = "artifacts"
os.makedirs(OUT_DIR, exist_ok=True)

df = pd.read_csv(DATA_CSV)   # must have reviews_text, user_sentiment (or rating), reviews_username, id, reviews_rating

# Basic cleaning
df['reviews_text'] = df['reviews_text'].fillna("").astype(str)
# Create binary labels: positive->1, others->0 (customize as needed)
df['label'] = df.get('user_sentiment', '').map({'positive':1, 'negative':0}).fillna(0).astype(int)

# Train TF-IDF + RandomForest
tfidf = TfidfVectorizer(max_features=20000, ngram_range=(1,2), min_df=3)
X = tfidf.fit_transform(df['reviews_text'])
y = df['label'].values
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)

rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)

# Save models
pickle.dump(tfidf, open(os.path.join(OUT_DIR, "tfidf_vectorizer.pkl"), "wb"))
pickle.dump(rf, open(os.path.join(OUT_DIR, "rf_model.pkl"), "wb"))

# Build user-item matrix
ui = df.pivot_table(index='reviews_username', columns='id', values='reviews_rating', aggfunc='mean').fillna(0)
pickle.dump(ui, open(os.path.join(OUT_DIR, "user_item_matrix.pkl"), "wb"))

# Item similarity
item_sim = cosine_similarity(ui.T)
item_sim_df = pd.DataFrame(item_sim, index=ui.columns, columns=ui.columns)
pickle.dump(item_sim_df, open(os.path.join(OUT_DIR, "item_similarity.pkl"), "wb"))

# Optional metadata (id->name)
meta = df.drop_duplicates(subset='id').set_index('id')[['name']].to_dict(orient='index')
pickle.dump(meta, open(os.path.join(OUT_DIR, "product_metadata.pkl"), "wb"))

print("Pickles saved to", OUT_DIR)
