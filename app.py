import streamlit as st
import pandas as pd
import re
import nltk
import joblib
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Download stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Clean input text
def clean_text(text):
    ps = PorterStemmer()
    text = re.sub('[^a-zA-Z]', ' ', str(text))
    text = text.lower()
    words = text.split()
    words = [ps.stem(word) for word in words if word not in stop_words]
    return ' '.join(words)

# Load pre-trained model and vectorizer
from transformers import pipeline

# Load a pre-trained BERT model for fake news detection
from transformers import pipeline

# Load BERT-based fake news classifier
classifier = pipeline("text-classification", model="mrm8488/bert-tiny-finetuned-fake-news")

# Get user input
text = input("Enter a news headline or article snippet:\n")

# Predict using the classifier
result = classifier(text)[0]
label = result['label']
score = result['score']

# Display the result
print(f"\n🧠 Prediction: {label}")
print(f"🔍 Confidence Score: {score:.2f}")

if label == "FAKE":
    print("❌ This article appears to be FAKE.")
else:
    print("✅ This article appears to be REAL.")


# Streamlit interface
def main():
    st.title("📰 Fake News Detector (Pretrained)")
    st.markdown("Paste a news article to detect if it's **Real** or **Fake**.")

    model, vectorizer = load_model()

    user_input = st.text_area("✏️ Paste your news article:")

    if st.button("🔍 Predict"):
        if not user_input.strip():
            st.warning("Please enter some text.")
        else:
            cleaned = clean_text(user_input)
            input_vectorized = vectorizer.transform([cleaned]).toarray()
            prediction = model.predict(input_vectorized)[0]

            if prediction == 1:
                st.error("🚨 This looks like **Fake News**.")
            else:
                st.success("✅ This appears to be **Real News**.")

if __name__ == "__main__":
    main()
