import streamlit as st
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import tensorflow as tf
import pickle

# Download necessary NLTK resources
nltk.download('stopwords')

# Initialize stopwords and stemmer
stop_words = set(stopwords.words('english'))
ps = PorterStemmer()

# Preprocessing function (text cleaning and stemming)
def clean_text(text):
    # Remove non-alphabetic characters
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    text = text.lower()
    words = text.split()
    
    # Remove stopwords and apply stemming
    words = [ps.stem(word) for word in words if word not in stop_words]
    
    # Return the cleaned text
    return ' '.join(words)

# Load the model (update the path to the model based on your setup)
@st.cache_resource
def load_model():
    try:
        # Replace with the correct model file (e.g., a Keras model or a pickle model)
        model = tf.keras.models.load_model('path_to_model/fake_news_model.h5')
        return model
    except:
        # If it's a pickle model, load it like this:
        model = pickle.load(open('path_to_model/fake_news_model.pkl', 'rb'))
        return model

# Streamlit Interface
def main():
    # Load the pre-trained model
    model = load_model()

    # Set the title and description of the app
    st.title("📰 Fake News Detector (Custom Model)")
    st.markdown("Paste a news article to detect if it's **Real** or **Fake**.")

    # Input field for news article
    user_input = st.text_area("✏️ Paste your news article here:")

    # Prediction button
    if st.button("🔍 Predict"):
        if not user_input.strip():
            st.warning("Please enter some text.")
        else:
            with st.spinner("Analyzing... Please wait."):

                # Preprocess the input text
                cleaned_input = clean_text(user_input)

                # If the model needs a vectorized input (e.g., TF-IDF or similar), vectorize the cleaned input here:
                # For example, if using TF-IDF vectorizer:
                # input_vectorized = vectorizer.transform([cleaned_input])

                # Make prediction using the model
                prediction = model.predict([cleaned_input])  # Modify this line if needed based on your model

                # Since the model's output could be either a probability or label, adjust the prediction handling
                if prediction == 1:
                    st.error("🚨 This looks like **Fake News**.")
                else:
                    st.success("✅ This appears to be **Real News**.")

    # Clear button to reset input field
    if st.button("❌ Clear Text"):
        st.experimental_rerun()

if __name__ == "__main__":
    main()
