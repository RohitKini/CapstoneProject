import os
import pickle
import pandas as pd

BASE = os.path.dirname(__file__)
ART = os.path.join(BASE, "artifacts")

def load_pickle(name):
    path = os.path.join(ART, name)
    return pickle.load(open(path, "rb"))

# Load all required data and models
try:
    user_item_matrix = load_pickle('user_item_matrix.pkl')
    item_similarity_df = load_pickle('item_similarity_df.pkl')
    rf_clf = load_pickle('rf_clf.pkl')
    tfidf_vectorizer = load_pickle('tfidf_vectorizer.pkl')
    df = load_pickle('df.pkl')

    if any(v is None for v in [user_item_matrix, item_similarity_df, rf_clf, tfidf_vectorizer, df]):
        raise ValueError("Required variables not found. Ensure pickles exist.")

    print("Successfully loaded models and data.")

except Exception as e:
    print(f"Error loading data and models: {e}")
    user_item_matrix = None
    item_similarity_df = None
    rf_clf = None
    tfidf_vectorizer = None
    df = None

def item_based_recommendations(item_name, user_item_matrix, item_similarity_df, n_recommendations=5):
    if item_name not in item_similarity_df.index:
        return []

    item_similarities = item_similarity_df.loc[item_name].drop(item_name, errors='ignore')
    users_who_rated_item = user_item_matrix.index[user_item_matrix[item_name] > 0]
    item_scores = {}

    for similar_item, similarity_score in item_similarities.items():
        if similarity_score <= 0:
            continue
        for user in users_who_rated_item:
            if user_item_matrix.loc[user, similar_item] > 0:
                if similar_item not in item_scores:
                    item_scores[similar_item] = 0
                item_scores[similar_item] += user_item_matrix.loc[user, similar_item] * similarity_score

    recommended_items = sorted(item_scores.items(), key=lambda x: x[1], reverse=True)
    return [item for item, score in recommended_items[:n_recommendations]]

# def get_recommendations_for_user(username, n_recommendations=5):
#     recommendations = []
#     predicted_sentiments_list = []

#     if user_item_matrix is None or item_similarity_df is None:
#         return [("Recommendation system not initialized.", "")]

#     if username not in user_item_matrix.index:
#         return [(f"User '{username}' not found.", "")]

#     user_rated_items = user_item_matrix.loc[username][user_item_matrix.loc[username] > 0].index.tolist()
#     if not user_rated_items:
#         return [("No items rated by user.", "")]

#     seed_item = user_rated_items[0]
#     recommended_items = item_based_recommendations(seed_item, user_item_matrix, item_similarity_df, n_recommendations)

#     if rf_clf is not None and tfidf_vectorizer is not None and df is not None:
#         for item_name in recommended_items:
#             item_reviews = df[df['name'] == item_name]['reviews_text_preprocessed']
#             if not item_reviews.empty:
#                 tfidf_reviews = tfidf_vectorizer.transform(item_reviews)
#                 preds = rf_clf.predict(tfidf_reviews)
#                 positive_pct = (preds == 'Positive').sum() / len(preds) * 100
#                 recommendations.append(item_name)
#                 predicted_sentiments_list.append(f"{positive_pct:.2f}% Positive")
#             else:
#                 recommendations.append(item_name)
#                 predicted_sentiments_list.append("N/A (No reviews)")
#     else:
#         predicted_sentiments_list = ["Sentiment N/A"] * len(recommended_items)

#     return list(zip(recommendations, predicted_sentiments_list))

def get_recommendations_for_user(username, n_recommendations=5):
    recommendations = []
    predicted_sentiments_list = []

    if user_item_matrix is None or item_similarity_df is None:
        return [("Recommendation system not initialized.", "")]

    if username not in user_item_matrix.index:
        return [(f"User '{username}' not found.", "")]

    user_rated_items = user_item_matrix.loc[username][user_item_matrix.loc[username] > 0].index.tolist()
    if not user_rated_items:
        return [("No items rated by user.", "")]

    seed_item = user_rated_items[0]
    recommended_items = item_based_recommendations(seed_item, user_item_matrix, item_similarity_df, n_recommendations)

    if rf_clf is not None and tfidf_vectorizer is not None and df is not None:
        for item_name in recommended_items:
            item_reviews = df[df['name'] == item_name]['reviews_text_preprocessed']
            if not item_reviews.empty:
                tfidf_reviews = tfidf_vectorizer.transform(item_reviews)
                preds = rf_clf.predict(tfidf_reviews)
                positive_pct = (preds == 'Positive').sum() / len(preds) * 100
                recommendations.append(item_name)
                predicted_sentiments_list.append(positive_pct)
            else:
                recommendations.append(item_name)
                predicted_sentiments_list.append(-1)  # use -1 for "no reviews"
    else:
        predicted_sentiments_list = [0] * len(recommended_items)

    # Combine and sort by sentiment score (descending)
    combined = list(zip(recommendations, predicted_sentiments_list))
    combined_sorted = sorted(combined, key=lambda x: x[1], reverse=True)

    # Format percentage nicely before returning
    final_output = [
        (item, f"{score:.2f}% Positive" if score >= 0 else "N/A (No reviews)")
        for item, score in combined_sorted
    ]

    return final_output
