import streamlit as st
import pickle
import pandas as pd

# Load vectorizer and model from the /mnt/data directory
with open('/mnt/data/vectorizer.pkl', 'rb') as vec_file:
    vectorizer = pickle.load(vec_file)

with open('/mnt/data/fake_news_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Streamlit UI
st.title("📰 Fake News Detector")

st.write("Enter news content below to predict whether it's Real or Fake.")

user_input = st.text_area("News Text", height=200)

if st.button("Predict"):
    if user_input.strip():
        try:
            input_vector = vectorizer.transform([user_input])
            prediction = model.predict(input_vector)[0]
            label = "🔴 FAKE" if prediction == 1 else "🟢 REAL"
            st.subheader(f"Prediction: {label}")
        except Exception as e:
            st.error(f"Error during prediction: {e}")
    else:
        st.warning("Please enter some text.")

# Optional: Show a few samples from CSV
if st.checkbox("Show sample dataset"):
    try:
        df = pd.read_csv("/mnt/data/small_file.csv")
        st.write(df.head())
    except Exception as e:
        st.error(f"Could not load dataset: {e}")
