import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import csr_matrix
import faiss
import altair as alt

st.set_page_config(page_title="🛍 Product Recommender", layout="wide")
st.title("🧠 Smart Product Recommendation System")

# Load Data
@st.cache_data
def load_data():
    df = pd.read_csv("ecommerce_customer_data_custom_ratios.csv")
    df = df.dropna(subset=['Customer ID', 'Product Category', 'Total Purchase Amount', 'Payment Method', 'Gender'])
    df['Customer ID'] = df['Customer ID'].astype(str)
    return df

df = load_data()

# Sidebar
rec_type = st.sidebar.radio("Recommendation Type", ["User-Based", "Content-Based"])
top_k = st.sidebar.slider("Number of Recommendations", 1, 20, 10)

# User-Based Functions
@st.cache_data
def prepare_user_based(df):
    matrix = df.pivot_table(index='Customer ID', columns='Product Category', values='Total Purchase Amount', aggfunc='sum', fill_value=0)
    sparse = csr_matrix(matrix.values)
    return matrix, sparse

def recommend_user_based(user_id, top_n=10):
    if user_id not in user_item_matrix.index:
        return []
    idx = user_item_matrix.index.get_loc(user_id)
    user_vec = user_item_sparse[idx]
    sim = cosine_similarity(user_vec, user_item_sparse).flatten()
    top_users = np.argsort(sim)[::-1][1:top_n + 20]
    items = user_item_matrix.iloc[top_users].sum(axis=0)
    already_bought = user_item_matrix.iloc[idx] > 0
    recs = items[~already_bought]
    return recs.sort_values(ascending=False).head(top_n)

# Content-Based Functions
@st.cache_resource
def prepare_content_based(df):
    df['metadata'] = df['Product Category'] + " " + df['Payment Method'] + " " + df['Gender']
    corpus = df['metadata'].fillna("").tolist()
    tfidf = TfidfVectorizer()
    tfidf_matrix = tfidf.fit_transform(corpus)

    num_features = tfidf_matrix.shape[1]
    n_components = min(100, num_features - 1)
    svd = TruncatedSVD(n_components=n_components)
    reduced = svd.fit_transform(tfidf_matrix)
    reduced = reduced / np.linalg.norm(reduced, axis=1, keepdims=True)

    d = reduced.shape[1]
    quantizer = faiss.IndexFlatIP(d)
    index = faiss.IndexIVFFlat(quantizer, d, 100, faiss.METRIC_INNER_PRODUCT)
    index.train(reduced.astype('float32'))
    index.add(reduced.astype('float32'))
    return reduced, index

def recommend_content_faiss(idx, reduced, index, top_k=10):
    distances, indices = index.search(reduced[idx].reshape(1, -1).astype('float32'), top_k + 1)
    return df.iloc[indices[0][1:]]

# Main UI
if rec_type == "User-Based":
    st.subheader("👤 User-Based Filtering")
    user_item_matrix, user_item_sparse = prepare_user_based(df)
    user_ids = user_item_matrix.index.tolist()
    selected_user = st.selectbox("Select Customer ID", user_ids)

    user_profile_df = df[df['Customer ID'] == selected_user]
    user_profile = user_profile_df.iloc[0]

    st.markdown("### 👤 Customer Profile")
    st.write({
        "Name": user_profile['Customer Name'],
        "Age": user_profile['Customer Age'],
        "Gender": user_profile['Gender'],
        "Churn": "Yes" if user_profile['Churn'] else "No"
    })

    st.markdown("### 📜 Purchase History")
    st.dataframe(user_profile_df[['Purchase Date', 'Product Category', 'Product Price', 'Payment Method', 'Returns']].sort_values('Purchase Date', ascending=False))

    st.markdown("### 📊 Top Categories Purchased")
    chart_data = user_profile_df['Product Category'].value_counts().reset_index()
    chart_data.columns = ['Product Category', 'Count']
    chart = alt.Chart(chart_data).mark_bar().encode(
        x='Product Category',
        y='Count',
        color='Product Category'
    ).properties(height=300)
    st.altair_chart(chart, use_container_width=True)

    if st.button("🔍 Recommend (User-Based)"):
        results = recommend_user_based(selected_user, top_k)
        if results is None or results.empty:
            st.warning("No recommendations found.")
        else:
            st.markdown("### 🛒 Top Recommendations")
            cols = st.columns(min(5, len(results)))
            for i, (cat, score) in enumerate(results.items()):
                emoji = "📦"
                with cols[i % len(cols)]:
                    st.markdown(f"""
                    <div style='background-color: #f0f8ff; padding: 10px; border-radius: 10px; text-align: center;'>
                        <strong>{emoji} {cat}</strong><br/>
                        Score: {score:.2f}
                    </div>""", unsafe_allow_html=True)

elif rec_type == "Content-Based":
    st.subheader("📦 Content-Based Filtering")
    reduced_matrix, faiss_index = prepare_content_based(df)
    df_display = df[['Product Category']].drop_duplicates().reset_index(drop=True)
    categories = df_display['Product Category'].tolist()
    selected = st.selectbox("Select Product Category", categories)

    if st.button("🔍 Recommend (Content-Based)"):
        sample_idx = df[df['Product Category'] == selected].index[0]
        selected_row = df.iloc[sample_idx]

        st.markdown("### 🔍 Selected Product Info")
        st.markdown(f"""
        <div style='background-color: #fff7e6; padding: 15px; border-radius: 10px;'>
            <strong>📦 Category:</strong> {selected_row['Product Category']}<br/>
            <strong>💳 Payment Method:</strong> {selected_row['Payment Method']}<br/>
            <strong>👤 Gender:</strong> {selected_row['Gender']}<br/>
        </div>
        """, unsafe_allow_html=True)

        result_df = recommend_content_faiss(sample_idx, reduced_matrix, faiss_index, top_k)
        if result_df.empty:
            st.warning("No similar products found.")
        else:
            st.markdown("### 🎯 Similar Products You Might Like")
            for idx, row in result_df.iterrows():
                bg_color = "#e3f2fd" if row['Payment Method'] == 'Credit Card' else "#f1f8e9"
                st.markdown(f"""
                <div style='background-color: {bg_color}; padding: 15px; border-radius: 12px; margin-bottom: 10px; box-shadow: 2px 2px 5px rgba(0,0,0,0.1);'>
                    <h4 style='color: #00796b; text-align: center;'>📦 {row['Product Category']}</h4>
                    <ul>
                        <li><strong>💰 Price:</strong> ₹{row['Product Price']}</li>
                        <li><strong>💳 Payment:</strong> {row['Payment Method']}</li>
                        <li><strong>👤 Gender:</strong> {row['Gender']}</li>
                        <li><strong>🔢 Quantity:</strong> {row['Quantity']}</li>
                        <li><strong>💵 Total:</strong> ₹{row['Total Purchase Amount']}</li>
                        <li><strong>🔁 Returns:</strong> {int(row['Returns']) if not pd.isna(row['Returns']) else 'N/A'}</li>
                        <li><strong>🕒 Date:</strong> {row['Purchase Date']}</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
