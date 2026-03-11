# 🛍️ Hybrid Customer Recommendation System

> A machine learning recommendation engine combining content-based and collaborative filtering to deliver personalized product suggestions — deployed as an interactive Streamlit web app.

![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python)
![Scikit-learn](https://img.shields.io/badge/ML-Scikit--learn-orange)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-red?logo=streamlit)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)

---

## 🚀 What It Does

Generic recommendations frustrate users and hurt engagement. This system tackles that by combining two powerful techniques:

- **Content-Based Filtering** — recommends products similar to what a user has liked before (using TF-IDF and cosine similarity)
- **Collaborative Filtering** — recommends products that similar users have liked (user–item interaction patterns)
- **Hybrid Approach** — blends both to overcome the cold-start problem and improve overall accuracy

---

## ✨ Features

- 🔍 **Personalized Recommendations** — tailored suggestions per user based on their history
- 🧮 **TF-IDF Vectorization** — converts product features into numerical representations
- 📐 **Cosine Similarity Scoring** — measures relevance between items and user profiles
- 🤝 **Collaborative Filtering** — leverages community behaviour to surface popular items
- ❄️ **Cold-Start Handling** — hybrid model reduces reliance on historical data
- 📊 **Interactive UI** — built with Streamlit for real-time recommendations

---

## 🏗️ How It Works

```
User Profile / History
        ↓
┌─────────────────────────────┐
│  Content-Based Filtering    │  → TF-IDF + Cosine Similarity
│  Collaborative Filtering    │  → User-Item Matrix
└─────────────────────────────┘
        ↓
   Hybrid Scoring Engine
        ↓
   Top-N Recommendations
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.9+ |
| ML | Scikit-learn, TF-IDF, Cosine Similarity |
| Data | Pandas, NumPy |
| UI | Streamlit |
| Dataset | E-commerce customer data (CSV) |

---

## ⚙️ Getting Started

### Installation
```bash
# Clone the repo
git clone https://github.com/Shraddhaaa05/customer-recommendation-system-.git
cd customer-recommendation-system-

# Install dependencies
pip install streamlit pandas numpy scikit-learn

# Run the app
streamlit run app.py
```

---

## 📁 Project Structure

```
├── app.py                                      # Streamlit web app
├── customer_recommendation.ipynb               # Full ML notebook (EDA + modelling)
├── ecommerce_customer_data_custom_ratios.csv   # Dataset
```

---

## 📊 Dataset

The project uses an e-commerce customer dataset containing user purchase history, product categories, and interaction ratios — enabling both content-based and collaborative filtering approaches.

---

## 🔍 Results

- Improved personalization accuracy by combining similarity scoring with interaction-based insights
- Reduced cold-start limitations through the hybrid blending approach
- Deployed as a real-time Streamlit application for interactive product discovery

---

## 👩‍💻 Author

**Shraddha Gidde**
- 🔗 [LinkedIn](https://www.linkedin.com/in/shraddha-gidde-063506242/)
- 💻 [GitHub](https://github.com/Shraddhaaa05)
