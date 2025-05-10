import streamlit as st
from transformers import pipeline

# Cache the model to avoid reloading on every prediction
@st.cache_resource
def load_model():
    # Load the pre-trained BERT model for fake news detection from Hugging Face
    return pipeline("text-classification", model="mrm8488/bert-tiny-finetuned-fake-news")

# Streamlit interface
def main():
    # Load the pre-trained model
    classifier = load_model()

    # Set the title and description of the app
    st.title("📰 Fake News Detector (BERT-based)")
    st.markdown("Paste a news article to detect if it's **Real** or **Fake**.")

    # Input field for news article
    user_input = st.text_area("✏️ Paste your news article here:")

    # Prediction button
    if st.button("🔍 Predict"):
        if not user_input.strip():
            st.warning("Please enter some text.")
        else:
            # Show loading spinner
            with st.spinner("Analyzing... Please wait."):

                # Get prediction from the BERT model
                result = classifier(user_input)[0]

                # Extract label (prediction) and confidence score
                label = result['label']
                score = result['score']

                # Display the prediction result
                st.write(f"🧠 **Prediction**: {label}")
                st.write(f"🔍 **Confidence Score**: {score:.2f}")

                # Display the result in a user-friendly format
                if label == "FAKE":
                    st.error("🚨 This looks like **Fake News**.")
                else:
                    st.success("✅ This appears to be **Real News**.")

    # Clear button to reset the input field
    if st.button("❌ Clear Text"):
        st.experimental_rerun()

if __name__ == "__main__":
    main()
