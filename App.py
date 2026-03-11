import streamlit as st
import pandas as pd
import numpy as np
import pickle
import faiss
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix

# ------------- CONFIGURATION ---------------
st.set_page_config(page_title="🧠 AI Product Recommender", layout="wide")
st.title("🛍️ AI-Powered Product Recommendation System")

# ------------- LOAD MODELS & DATA ---------------
@st.cache_data
def load_all():
    df = pd.read_csv("ecommerce_customer_data_custom_ratios.csv")
    with open("models/user_item_matrix.pkl", "rb") as f:
        user_item_matrix = pickle.load(f)
    with open("models/user_item_sparse.pkl", "rb") as f:
        user_item_sparse = pickle.load(f)
    with open("models/reduced_matrix.pkl", "rb") as f:
        reduced_matrix = pickle.load(f)
    faiss_index = faiss.read_index("models/faiss_index.index")
    return df, user_item_matrix, user_item_sparse, reduced_matrix, faiss_index

# ------------- RECOMMENDATION LOGIC ---------------
def recommend_user_based(user_id, user_item_matrix, user_item_sparse, top_n=5):
    if user_id not in user_item_matrix.index:
        return pd.Series(dtype=float)
    idx = user_item_matrix.index.get_loc(user_id)
    user_vec = user_item_sparse[idx]
    sim = cosine_similarity(user_vec, user_item_sparse).flatten()
    top_users = sim.argsort()[::-1][1:top_n + 20]
    items = user_item_matrix.iloc[top_users].sum(axis=0)
    already_bought = user_item_matrix.iloc[idx] > 0
    recs = items[~already_bought]
    return recs.sort_values(ascending=False).head(top_n)

def recommend_content_faiss(item_idx, reduced_matrix, faiss_index, df, top_k=5):
    distances, indices = faiss_index.search(reduced_matrix[item_idx].reshape(1, -1).astype("float32"), top_k + 1)
    return df.iloc[indices[0][1:]]

def recommend_hybrid(user_id, item_idx, df, user_item_matrix, user_item_sparse, reduced_matrix, faiss_index, user_weight=0.6, top_k=5):
    user_scores = recommend_user_based(user_id, user_item_matrix, user_item_sparse, top_k + 20)
    content_scores_df = recommend_content_faiss(item_idx, reduced_matrix, faiss_index, df, top_k + 20)
    content_scores = content_scores_df['Product Category'].value_counts()
    hybrid_df = pd.DataFrame(index=list(set(user_scores.index).union(content_scores.index)))
    hybrid_df['user'] = user_scores
    hybrid_df['content'] = content_scores
    hybrid_df = hybrid_df.fillna(0)
    hybrid_df['score'] = hybrid_df['user'] * user_weight + hybrid_df['content'] * (1 - user_weight)
    top_cats = hybrid_df.sort_values(by='score', ascending=False).head(top_k).reset_index()
    top_cats = top_cats.rename(columns={'index': 'Product Category'})
    merged = pd.merge(top_cats, df[['Product Category', 'Customer Name', 'Gender', 'Customer Age']], on='Product Category', how='left')
    return merged.drop_duplicates(subset='Product Category')

# ------------- MAIN APP START ---------------
df, user_item_matrix, user_item_sparse, reduced_matrix, faiss_index = load_all()

rec_type = st.sidebar.radio("📌 Choose Recommendation Mode:", ["User-Based", "Content-Based", "Hybrid", "Evaluation"])
top_k = st.sidebar.slider("🔢 Number of Recommendations", 1, 20, 5)
sample_size = st.sidebar.slider("📊 Evaluation Sample Size", 10, 200, 50)
filter_gender = st.sidebar.selectbox("🧍 Filter by Gender", ["All", "Male", "Female"])
filter_age = st.sidebar.slider("🎂 Max Customer Age", 18, 80, 80)
# ----------------- USER-BASED ------------------
if rec_type == "User-Based":
    st.header("👤 Personalized Recommendations (User-Based)")
    user_ids = user_item_matrix.index.tolist()
    selected_user = st.selectbox("Select a Customer ID:", user_ids)

    user_df = df[df["Customer ID"] == selected_user]
    st.markdown("#### 👨‍💼 Customer Profile")
    st.dataframe(user_df[['Customer Name', 'Gender', 'Customer Age', 'Churn']].drop_duplicates(), use_container_width=True)

    st.markdown("#### 🛒 Purchase History")
    st.dataframe(user_df[['Purchase Date', 'Product Category', 'Product Price', 'Payment Method']], use_container_width=True)

    if st.button("🚀 Show Recommendations"):
        recs = recommend_user_based(selected_user, user_item_matrix, user_item_sparse, top_k)
        if recs.empty:
            st.warning("No new recommendations available for this user.")
        else:
            merged = pd.merge(recs.reset_index(), df[['Product Category', 'Customer Name', 'Gender', 'Customer Age']], on='Product Category', how='left')
            merged = merged.drop_duplicates(subset='Product Category')
            merged = merged.rename(columns={0: 'Score'})[['Customer Name', 'Product Category', 'Gender', 'Customer Age', 'Score']]
            if filter_gender != "All":
                merged = merged[merged['Gender'] == filter_gender]
            merged = merged[merged['Customer Age'] <= filter_age]
            st.success(f"Top {top_k} Recommended Categories for Customer {selected_user}")
            st.dataframe(merged, use_container_width=True)
            st.download_button("📥 Download Recommendations", merged.to_csv(index=False), file_name="user_based_recommendations.csv")
            if rec_type == "User-Based":
                st.header("👤 User-Based Recommendations")
    user_ids = user_item_matrix.index.tolist()
    selected_user = st.selectbox("Select a Customer ID:", user_ids, key="user_userbased")

    if st.button("🧠 Recommend (User-Based)"):
        recs = recommend_user_based(selected_user, user_item_matrix, user_item_sparse, top_k)
        if recs.empty:
            st.warning("No recommendations found for this user.")
        else:
            merged = pd.merge(recs.reset_index(), df[['Product Category', 'Customer Name', 'Gender', 'Customer Age']],
                              on='Product Category', how='left').drop_duplicates(subset='Product Category')
            merged = merged.rename(columns={0: 'Score'})

            # Apply filters
            if filter_gender != "All":
                merged = merged[merged['Gender'] == filter_gender]
            merged = merged[merged['Customer Age'] <= filter_age]

            st.dataframe(merged[['Customer Name', 'Product Category', 'Gender', 'Customer Age', 'Score']], use_container_width=True)

            # ✅ Bar chart for recommended Product Categories
            st.markdown("### 📊 Recommended Product Category Count")
            category_counts = merged['Product Category'].value_counts()
            fig1, ax1 = plt.subplots()
            category_counts.plot(kind='barh', ax=ax1, color='lightgreen')
            ax1.set_xlabel("Count")
            ax1.set_ylabel("Product Category")
            st.pyplot(fig1)

# ----------------- CONTENT-BASED ------------------
elif rec_type == "Content-Based":
    st.header("📦 Similar Products Recommender (Content-Based)")
    product_categories = df['Product Category'].unique().tolist()
    selected_category = st.selectbox("Select a Product Category:", product_categories)

    if st.button("🔍 Find Similar Products"):
        idx = df[df["Product Category"] == selected_category].index[0]
        results = recommend_content_faiss(idx, reduced_matrix, faiss_index, df, top_k)
        results = results[['Customer Name', 'Product Category', 'Payment Method', 'Gender', 'Customer Age', 'Product Price']]
        if filter_gender != "All":
            results = results[results['Gender'] == filter_gender]
        results = results[results['Customer Age'] <= filter_age]
        st.success(f"Top {top_k} Products Similar to '{selected_category}'")
        st.dataframe(results, use_container_width=True)
        st.download_button("📥 Download Recommendations", results.to_csv(index=False), file_name="content_based_recommendations.csv")
        # ✅ Visualization: Distribution of product prices
        st.markdown("### 💰 Price Distribution of Similar Products")
        fig2, ax2 = plt.subplots()
        results['Product Price'].plot(kind='hist', bins=10, ax=ax2, color='skyblue', edgecolor='black')
        ax2.set_xlabel("Product Price")
        ax2.set_ylabel("Frequency")
        st.pyplot(fig2)

# ------------- HYBRID RECOMMENDER ---------------
if rec_type == "Hybrid":
    st.header("🤝 Hybrid Recommendations")
    selected_user = st.selectbox("Select Customer ID:", user_item_matrix.index.tolist())
    selected_category = st.selectbox("Select a Product Category:", df['Product Category'].unique())

    if st.button("🔀 Recommend (Hybrid)"):
        idx = df[df['Product Category'] == selected_category].index[0]
        hybrid_results = recommend_hybrid(selected_user, idx, df, user_item_matrix, user_item_sparse, reduced_matrix, faiss_index, top_k=top_k)

        st.dataframe(hybrid_results[['Customer Name', 'Product Category', 'Gender', 'Customer Age', 'score']], use_container_width=True)

        st.markdown("### 📊 Score Visualization")
        fig, ax = plt.subplots()
        ax.barh(hybrid_results['Product Category'], hybrid_results['score'], color='skyblue')
        plt.xlabel("Hybrid Score")
        plt.ylabel("Product Category")
        st.pyplot(fig)

# ------------- EVALUATION TAB ---------------
elif rec_type == "Evaluation":
    st.header("📈 Recommendation Evaluation")

    def get_ground_truth_matrix(df):
        return df.groupby(['Customer ID', 'Product Category']).size().unstack(fill_value=0)

    def evaluate_top_k(ground_truth, predictions, k):
        precision, recall = [], []
        for user, actual_items in ground_truth.iterrows():
            actual = set(actual_items[actual_items > 0].index)
            pred = set(predictions.get(user, [])[:k])
            tp = len(actual & pred)
            precision.append(tp / k if k > 0 else 0)
            recall.append(tp / len(actual) if len(actual) > 0 else 0)
        precision_avg = np.mean(precision)
        recall_avg = np.mean(recall)
        f1 = 2 * (precision_avg * recall_avg) / (precision_avg + recall_avg + 1e-6)
        return {"Precision@K": precision_avg, "Recall@K": recall_avg, "F1 Score": f1}

    if st.button("▶️ Run Evaluation"):
        truth_matrix = get_ground_truth_matrix(df)
        user_ids = truth_matrix.index.tolist()[:sample_size]

        progress = st.progress(0)
        rec_user, rec_content, rec_hybrid = {}, {}, {}

        for i, uid in enumerate(user_ids):
            if uid not in user_item_matrix.index: continue
            rec_user[uid] = recommend_user_based(uid, user_item_matrix, user_item_sparse, top_k).index.tolist()
            try:
                idx = df[df['Customer ID'] == uid].index[0]
                rec_content[uid] = recommend_content_faiss(idx, reduced_matrix, faiss_index, df, top_k)['Product Category'].tolist()
                rec_hybrid[uid] = recommend_hybrid(uid, idx, df, user_item_matrix, user_item_sparse, reduced_matrix, faiss_index, top_k)['Product Category'].tolist()
            except: continue
            progress.progress((i + 1) / len(user_ids))

        metrics_user = evaluate_top_k(truth_matrix.loc[user_ids], rec_user, top_k)
        metrics_content = evaluate_top_k(truth_matrix.loc[user_ids], rec_content, top_k)
        metrics_hybrid = evaluate_top_k(truth_matrix.loc[user_ids], rec_hybrid, top_k)

        results = pd.DataFrame({
            "User-Based": metrics_user,
            "Content-Based": metrics_content,
            "Hybrid": metrics_hybrid
        }).T

        st.markdown("### 📋 Evaluation Metrics")
        st.dataframe(results.style.format(precision=4))

        st.markdown("### 📉 Visual Comparison")
        fig, ax = plt.subplots(figsize=(8, 4))
        results.plot(kind="bar", ax=ax, colormap='viridis')
        plt.title("Model Performance Comparison")
        plt.ylabel("Score")
        st.pyplot(fig)
