import streamlit as st
import joblib
import os
import re

# Must be FIRST Streamlit command
st.set_page_config(page_title="Fake News Detector", page_icon="📰")

@st.cache_resource
def load_model():
    # Path to YOUR saved files (from notebook)
    model_path = "../models/fake_news_detector.pkl"
    vectorizer_path = "../models/tfidf_vectorizer.pkl"
    
    if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
        st.error("❌ Model files not found in ../Model/. Run notebook first!")
        st.stop()
    
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    return model, vectorizer

# Load model and vectorizer
model, vectorizer = load_model()

def clean_text(text):
    """Simple text cleaning like your notebook"""
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    return ' '.join(text.split())

st.title("📰 Fake News Detection System")
st.markdown("**86.1% accuracy** on **LIAR + WELFAKE + ISOT** (75K articles)")

# Sidebar info
with st.sidebar:
    st.info("✅ **Test these:**")
    st.code('"Earth is flat, NASA confirms!" → FAKE')
    st.code('"Reuters: Putin ready for US dialogue" → REAL')

# Main input
news = st.text_area("Enter News Article:", height=200, 
                   placeholder="Paste news text here...")

if st.button("🔍 Detect News", type="primary"):
    if news.strip() == "":
        st.warning("⚠️ Please enter news text")
    else:
        # Clean and predict
        cleaned = clean_text(news)
        vec = vectorizer.transform([cleaned])
        prediction = model.predict(vec)[0]
        confidence = abs(model.decision_function(vec)[0])
        
        st.subheader("📊 **Result**")
        
        col1, col2 = st.columns([3, 1])
        with col1:
            if prediction == 1:
                st.success("✅ **REAL NEWS**")
            else:
                st.error("🚨 **FAKE NEWS**")
        with col2:
            st.metric("Confidence", f"{confidence:.1%}")
        
        st.caption(f"Processed: {len(cleaned.split())} words")

st.markdown("---")
st.caption("🎓 Fake News Detection | B.Tech AI/ML Project")
