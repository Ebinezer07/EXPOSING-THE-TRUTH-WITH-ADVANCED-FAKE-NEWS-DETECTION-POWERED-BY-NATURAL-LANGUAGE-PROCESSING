import streamlit as st
import pandas as pd
import pickle

# Load vectorizer and model from uploaded files
@st.cache_resource
def load_model():
    with open("/mnt/data/vectorizer.pkl", "rb") as vec_file:
        vectorizer = pickle.load(vec_file)
    with open("/mnt/data/fake_news_model.pkl", "rb") as model_file:
        model = pickle.load(model_file)
    return vectorizer, model

vectorizer, model = load_model()

st.title("📰 Fake News Detector")
st.markdown("Detect whether a news article is **real** or **fake** using a trained ML model.")

# --- Single text prediction ---
st.header("🔍 Single Article Prediction")

user_input = st.text_area("Enter the news content here:", height=200)

if st.button("Predict"):
    if user_input.strip():
        try:
            input_vector = vectorizer.transform([user_input])
            prediction = model.predict(input_vector)[0]
            label = "🔴 FAKE" if prediction == 1 else "🟢 REAL"
            st.success(f"Prediction: {label}")
        except Exception as e:
            st.error(f"Prediction error: {e}")
    else:
        st.warning("Please enter some text.")

# --- Batch CSV prediction ---
st.header("📄 Batch Prediction from CSV")
st.markdown("Upload a CSV file with a column of news content for batch predictions.")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        text_column = st.selectbox("Select the column containing news text:", df.columns)

        if st.button("Run Batch Prediction"):
            text_data = df[text_column].astype(str)
            X_vec = vectorizer.transform(text_data)
            predictions = model.predict(X_vec)
            df["Prediction"] = ["FAKE" if p == 1 else "REAL" for p in predictions]
            st.success("Batch prediction completed!")
            st.write(df.head())

            # Optional: download link
            csv = df.to_csv(index=False)
            st.download_button("Download Results CSV", csv, "predictions.csv", "text/csv")

    except Exception as e:
        st.error(f"Failed to process uploaded file: {e}")
