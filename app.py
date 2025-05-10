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
import pickle

# Load the trained model and vectorizer
model = pickle.load(open('fake_news_model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

# Get user input
text = [input("Enter your news text:\n")]

# Transform the input text using the vectorizer
vect = vectorizer.transform(text)

# Predict using the model
result = model.predict(vect)

# Display the result
print(result)


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
