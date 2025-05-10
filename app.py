# Load small local CSV file with validation
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("small_file.csv")
        # Attempt to auto-detect correct column names
        expected_cols = df.columns.str.lower().tolist()

        # Map similar column names
        text_col = next((col for col in df.columns if col.lower() == 'text'), None)
        label_col = next((col for col in df.columns if col.lower() == 'label'), None)

        # If not found, raise an error
        if not text_col or not label_col:
            st.error(f"❌ CSV must contain 'text' and 'label' columns. Found: {df.columns.tolist()}")
            return pd.DataFrame()  # Return empty DataFrame

        # Rename columns to expected names
        df = df.rename(columns={text_col: "text", label_col: "label"})
        return df

    except Exception as e:
        st.error(f"❌ Failed to load CSV: {e}")
        return pd.DataFrame()

# Train model with cleaned column names
@st.cache_resource
def train_model(df):
    if df.empty:
        return None, None, 0.0

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
