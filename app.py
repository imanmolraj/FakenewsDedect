import re
import joblib
import requests
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import shap
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibrationDisplay
from sklearn.metrics import ConfusionMatrixDisplay
import nltk

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')


# Define the custom model class FIRST
class BalancedLogisticRegression(LogisticRegression):
    def fit(self, X, y, sample_weight=None):
        if sample_weight is None:
            class_weights = {0: len(y)/sum(y==0), 1: len(y)/sum(y==1)}
            sample_weight = [class_weights[c] for c in y]
        return super().fit(X, y, sample_weight)

# Download NLTK resources
nltk.download(['punkt', 'stopwords'])

# Load models
tfidf = joblib.load('tfidf_vectorizer.pkl')
model = joblib.load('news_model.pkl')

# Configure API
API_KEY = 'YOUR_GOOGLE_FACTCHECK_API_KEY'
API_ENDPOINT = "https://factchecktools.googleapis.com/v1alpha1/claims:search"

# Text preprocessing
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def preprocess(text):
    text = re.sub(r'[^a-zA-Z\s]', '', str(text), flags=re.I).lower()
    tokens = nltk.word_tokenize(text)
    return ' '.join([stemmer.stem(word) for word in tokens 
                    if word not in stop_words and len(word) > 2])

def check_fact_claims(text):
    params = {'query': text, 'key': API_KEY}
    try:
        response = requests.get(API_ENDPOINT, params=params)
        response.raise_for_status()
        claims = response.json().get('claims', [])
        return claims if claims else None
    except Exception as e:
        return f"API Error: {str(e)}"

def get_source_credibility(text):
    verified_sources = ['reuters', 'associated press', 'bbc']
    return "Verified Source" if any(source in text.lower() for source in verified_sources) else "Unverified Source"

# Streamlit UI
st.title("🕵️♂️ Fake News Detector")
st.subheader("AI-Powered Verification with Fact-Checking")

input_text = st.text_area("Paste news article here:", height=200)

if st.button("Analyze Article"):
    if not input_text.strip():
        st.error("Please enter a news article!")
    else:
        # Preprocessing
        clean_text = preprocess(input_text)
        
        # Transform and predict
        vector = tfidf.transform([clean_text])
        pred = model.predict(vector)[0]
        proba = model.predict_proba(vector)[0][1]
        
        # Source verification
        source_status = get_source_credibility(input_text)
        
        # Display results
        st.markdown(f"**Source:** {source_status}")
        st.markdown(f"**Prediction:** {'FAKE NEWS 🔴' if pred == 0 else 'REAL NEWS 🟢'}")
        st.markdown(f"**Confidence:** {proba*100:.2f}%")
        
        # Explain features
        st.subheader("Top Predictive Features")
        feature_names = tfidf.get_feature_names_out()
        coefs = model.coef_[0]

        top_features = sorted(zip(feature_names, coefs), 
                              key=lambda x: abs(x[1]), 
                              reverse=True)[:5]
        
        for feat, weight in top_features:
            st.write(f"- {'🚩' if weight <0 else '✅'} {feat}: {abs(weight):.2f}")
        
        # SHAP explanation
        st.subheader("Prediction Explanation")
        explainer = shap.LinearExplainer(model, model.coef_)
        shap_values = explainer.shap_values(vector)
        fig, ax = plt.subplots()
        shap.plots.waterfall(shap_values[0], max_display=10, show=False)
        st.pyplot(fig)
        
        # Fact-check results
        st.subheader("Fact-Check Findings")
        fact_check = check_fact_claims(input_text)
        if isinstance(fact_check, str):
            st.write(fact_check)
        elif fact_check:
            for claim in fact_check[:3]:
                st.write(f"- **Claim:** {claim.get('text', 'N/A')}")
                st.write(f"  **By:** {claim.get('claimant', 'Unknown')}")
                if claim.get('claimReview'):
                    review = claim['claimReview'][0]
                    st.write(f"  **Verdict:** {review.get('textualRating', 'N/A')}")
                    st.write(f"  **Source:** {review.get('publisher', {}).get('name', 'N/A')}")
                    st.write(f"  **URL:** {review.get('url', 'N/A')}")
        else:
            st.write("No related fact-checks found")
