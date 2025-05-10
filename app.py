import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import hashlib

# Function to create a hash of the file to detect changes and clear cache
def get_file_hash(file_path):
    with open(file_path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()

# Load data with caching
@st.cache_data
def load_data(file_hash):
    try:
        df = pd.read_csv("small_file.csv")
        
        # Check if the necessary columns are present
        expected_cols = df.columns.str.lower().tolist()
        text_col = next((col for col in df.columns if col.lower() == 'text'), None)
        label_col = next((col for col in df.columns if col.lower() == 'label'), None)

        if not text_col or not label_col:
            st.error(f"❌ CSV must contain 'text' and 'label' columns. Found: {df.columns.tolist()}")
            return pd.DataFrame()  # Return empty DataFrame

        # Rename columns to expected names
        df = df.rename(columns={text_col: "text", label_col: "label"})
        return df

    except Exception as e:
        st.error(f"❌ Failed to load CSV: {e}")
        return pd.DataFrame()

# Train model with caching, avoid caching large models or data
@st.cache_resource
def train_model(df):
    if df.empty:
        return None, None, 0.0

    df['cleaned_text'] = df['text'].apply(clean_text)  # Assuming clean_text function is defined
    X = df['cleaned_text']
    y = df['label']

    vectorizer = TfidfVectorizer(max_features=5000)
    X_vectorized = vectorizer.fit_transform(X).toarray()

    X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.2, random_state=42)

    model = LogisticRegression()
    model.fit(X_train, y_train)

    accuracy = accuracy_score(y_test, model.predict(X_test))
    return model, vectorizer, accuracy

# Streamlit UI
def app():
    # Get the current hash of the file
    file_hash = get_file_hash("small_file.csv")
    
    # Clear cache when the file changes
    st.session_state.file_hash = file_hash
    
    # Load the data
    df = load_data(st.session_state.file_hash)

    if df.empty:
        return

    # Train the model
    model, vectorizer, accuracy = train_model(df)
    
    if model:
        st.write(f"Model Accuracy: {accuracy * 100:.2f}%")

# Run the app
if __name__ == "__main__":
    app()
