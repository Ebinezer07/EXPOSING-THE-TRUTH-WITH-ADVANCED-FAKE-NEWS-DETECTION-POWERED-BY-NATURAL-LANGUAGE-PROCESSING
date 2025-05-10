import streamlit as st
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Download stopwords once
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Text cleaning function
def clean_text(text):
    ps = PorterStemmer()
    text = re.sub('[^a-zA-Z]', ' ', str(text))
    text = text.lower()
    words = text.split()
    words = [ps.stem(word) for word in words if word not in stop_words]
    return ' '.join(words)

# Load small local CSV file
@st.cache_data
def load_data():
    df = pd.read_csv("small_file.csv")  # Make sure this file exists in the same directory
    return df

# Train model with cached result
@st.cache_resource
def train_model(df):
    df['cleaned_text'] = df['text'].apply(clean_text)
    X = df['cleaned_text']
    y = df['label']

    vectorizer = TfidfVectorizer(max_features=5000)
    X_vectorized = vectorizer.fit_transform(X).toarray()

    X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.2, random_state=42)

    model = LogisticRegression()
    model.fit(X_train, y_train)

    accuracy = accuracy_score(y_test, model.predict(X_test))
    return model, vectorizer, accuracy

# Streamlit App
def main():
    st.title("📰 Fake News Detection Using NLP")
    st.markdown("Enter a news article below to check if it's **Real** or **Fake**.")

    with st.spinner("📊 Loading data and training model..."):
        df = load_data()
        if df.empty:
            st.error("❌ Failed to load data.")
            return

        model, vectorizer, accuracy = train_model(df)
        if model is None:
            st.error("❌ Model training failed.")
            return

    st.success(f"✅ Model trained with **{accuracy:.2%}** accuracy.")

    user_input = st.text_area("✏️ Paste your news article here:")

    if st.button("🔍 Predict"):
        if not user_input.strip():
            st.warning("⚠️ Please enter some text.")
        else:
            cleaned_input = clean_text(user_input)
            input_vectorized = vectorizer.transform([cleaned_input]).toarray()
            prediction = model.predict(input_vectorized)[0]

            if prediction == 1:
                st.error("🚨 This looks like **Fake News**.")
            else:
                st.success("✅ This appears to be **Real News**.")

if __name__ == "__main__":
    main()
