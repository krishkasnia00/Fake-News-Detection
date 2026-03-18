import streamlit as st
import joblib
import re
import nltk
import os
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# --- STEP 1: MUST BE THE FIRST STREAMLIT COMMAND ---
st.set_page_config(page_title="Fake News Detector", layout="centered")

# --- 2. Setup NLTK (Quietly) ---
@st.cache_resource
def download_nltk_data():
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)

download_nltk_data()
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# --- 3. Load Assets with Absolute Paths ---
@st.cache_resource
def load_assets():
    base_path = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_path, 'random_forest_model.pkl')
    vectorizer_path = os.path.join(base_path, 'tfidf_vectorizer.pkl')
    
    if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
        return None, None, f"Missing files in: {base_path}"
        
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    return model, vectorizer, None

model, vectorizer, error_msg = load_assets()

# Check for loading errors after config is set
if error_msg:
    st.error(error_msg)
    st.stop()

# --- 4. Text Preprocessing ---
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    words = text.split()
    words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
    return " ".join(words)

# --- 5. UI Layout ---
st.title("📰 Fake News Detection System")
st.markdown("Determine if an article is **Real** or **Fake** using AI.")

user_input = st.text_area("Paste article text:", height=250)

if st.button("Analyze Article", type="primary"):
    if user_input.strip():
        cleaned_text = clean_text(user_input)
        vectorized_input = vectorizer.transform([cleaned_text])
        
        prediction = model.predict(vectorized_input)[0]
        prob = model.predict_proba(vectorized_input)[0]
        
        if prediction == 1:
            st.success(f"###  Likely REAL News (Confidence: {prob[1]:.2%})")
        else:
            st.error(f"###  Likely FAKE News (Confidence: {prob[0]:.2%})")
    else:
        st.warning("Please paste some text first.")