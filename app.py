import streamlit as st
import pickle
import pandas as pd

# Load vectorizer and model
@st.cache_resource
def load_model_and_vectorizer():
    with open("vectorizer.pkl", "rb") as vec_file:
        vectorizer = pickle.load(vec_file)
    with open("fake_news_model.pkl", "rb") as model_file:
        model = pickle.load(model_file)
    return vectorizer, model

vectorizer, model = load_model_and_vectorizer()

# Streamlit UI
st.set_page_config(page_title="Fake News Detector", layout="centered")
st.title("📰 Fake News Detector")
st.markdown("Detect whether a news article is **Real** or **Fake** using NLP & Machine Learning.")

# --- Text Input Section ---
st.subheader("🔍 Predict a Single News Article")

user_input = st.text_area("Enter News Content Here", height=150)

if st.button("Predict"):
    if user_input.strip():
        cleaned_text = user_input.lower()
        input_vector = vectorizer.transform([cleaned_text])
        prediction = model.predict(input_vector)[0]
        label = "🔴 FAKE" if prediction == 1 else "🟢 REAL"
        st.success(f"Prediction: {label}")
    else:
        st.warning("⚠️ Please enter some text.")

# --- CSV Upload Section ---
st.subheader("📄 Batch Prediction from CSV")
st.markdown("Upload a CSV with a column of news articles to predict in bulk.")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        column = st.selectbox("Select text column", df.columns)

        if st.button("Run Batch Prediction"):
            text_data = df[column].astype(str).str.lower()
            X = vectorizer.transform(text_data)
            preds = model.predict(X)
            df["Prediction"] = ["FAKE" if p == 1 else "REAL" for p in preds]

            st.success("✅ Predictions complete!")
            st.write(df.head())

            # Option to download results
            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button("Download Predictions", csv, "fake_news_predictions.csv", "text/csv")

    except Exception as e:
        st.error(f"Error: {e}")
