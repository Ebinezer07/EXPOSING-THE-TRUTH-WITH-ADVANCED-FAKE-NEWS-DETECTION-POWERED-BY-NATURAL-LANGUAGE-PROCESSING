import streamlit as st
import pandas as pd
import joblib

# Load vectorizer and model
@st.cache_resource
def load_artifacts():
    vectorizer = joblib.load("vectorizer.pkl")
    model = joblib.load("fake_news_model.pkl")
    return vectorizer, model

vectorizer, model = load_artifacts()

st.title("📰 Fake News Detector")
st.markdown("Check if a news article is **REAL** or **FAKE** using Machine Learning!")

# Single Prediction
st.subheader("🔍 Predict One Article")
user_input = st.text_area("Enter news content")

if st.button("Predict"):
    if user_input.strip():
        vec = vectorizer.transform([user_input.lower()])
        pred = model.predict(vec)[0]
        st.success("🟢 REAL" if pred == 0 else "🔴 FAKE")
    else:
        st.warning("Please enter some text.")

# Batch Prediction
st.subheader("📄 Batch Prediction")
uploaded_file = st.file_uploader("Upload a CSV with a 'text' column", type="csv")

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        if "text" not in df.columns:
            st.error("CSV must contain a 'text' column.")
        else:
            text_data = df["text"].astype(str).str.lower()
            vec_data = vectorizer.transform(text_data)
            preds = model.predict(vec_data)
            df["Prediction"] = ["REAL" if p == 0 else "FAKE" for p in preds]
            st.dataframe(df.head())
            st.download_button("Download Results", df.to_csv(index=False), "predictions.csv", "text/csv")
    except Exception as e:
        st.error(f"Error processing file: {e}")
