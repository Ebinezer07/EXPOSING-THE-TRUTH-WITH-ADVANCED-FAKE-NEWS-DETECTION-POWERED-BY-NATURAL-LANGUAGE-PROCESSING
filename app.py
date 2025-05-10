import streamlit as st
import pickle
import pandas as pd

# Load vectorizer and model
with open('vectorizer.pkl', 'rb') as vec_file:
    vectorizer = pickle.load(vec_file)

with open('fake_news_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Streamlit UI
st.title("📰 Fake News Detector")

st.write("Enter news content below to predict whether it's Real or Fake.")

user_input = st.text_area("News Text", height=200)

if st.button("Predict"):
    if user_input.strip():
        input_vector = vectorizer.transform([user_input])
        prediction = model.predict(input_vector)[0]
        label = "🔴 FAKE" if prediction == 1 else "🟢 REAL"
        st.subheader(f"Prediction: {label}")
    else:
        st.warning("Please enter some text.")

# Optional: Show a few samples from CSV
if st.checkbox("Show sample dataset"):
    df = pd.read_csv("small_file.csv")
    st.write(df.head())
