import streamlit as st
import torch
import torch.nn.functional as F
from transformers import BertTokenizer, BertForSequenceClassification
import os

# Page config
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="🔍",
    layout="centered"
)

# Title
st.title("🔍 Fake News Detector")
st.subheader("Powered by BERT AI Model")
st.markdown("---")

# Model load karo
@st.cache_resource
def load_model():
    model_path = "fake_news_model"
    tokenizer = BertTokenizer.from_pretrained(model_path)
    model = BertForSequenceClassification.from_pretrained(model_path)
    model.eval()
    return tokenizer, model

# Loading spinner
with st.spinner("Loading AI Model..."):
    tokenizer, model = load_model()

st.success("Model ready!")

# Predict function
def predict(text):
    inputs = tokenizer(
        text,
        return_tensors='pt',
        truncation=True,
        padding=True,
        max_length=128
    )
    with torch.no_grad():
        outputs = model(**inputs)
        probs = F.softmax(outputs.logits, dim=1)
    
    fake_prob = probs[0][1].item() * 100
    real_prob = probs[0][0].item() * 100
    return fake_prob, real_prob

# Input section
st.markdown("### 📰 Enter News Headline or Article:")
user_input = st.text_area("", height=150, 
    placeholder="Paste your news headline here...")

# Sample buttons
st.markdown("**Or try these samples:**")
col1, col2 = st.columns(2)

with col1:
    if st.button("🇮🇳 PM Kisan Scheme"):
        user_input = "PM Modi launches PM Kisan scheme for farmers worth 6000 rupees annually"
    if st.button("💊 Fake Health Claim"):
        user_input = "Drinking cow urine cures cancer and COVID-19 completely"

with col2:
    if st.button("🏅 Olympics News"):
        user_input = "India wins gold medal at Paris Olympics 2024 in shooting"
    if st.button("💰 WhatsApp Scam"):
        user_input = "Bill Gates will give you 1 crore rupees if you share this message"

# Detect button
if st.button("🔍 DETECT NOW", type="primary"):
    if user_input.strip() == "":
        st.warning("Please enter some text first!")
    else:
        with st.spinner("Analyzing..."):
            fake_prob, real_prob = predict(user_input)
        
        st.markdown("---")
        st.markdown("### Result:")
        
        if real_prob > fake_prob:
            st.success(f"✅ REAL NEWS — Confidence: {real_prob:.1f}%")
        else:
            st.error(f"🚨 FAKE NEWS — Confidence: {fake_prob:.1f}%")
        
        # Progress bars
        st.markdown("**Probability Breakdown:**")
        st.markdown("🔴 Fake News")
        st.progress(fake_prob/100)
        st.caption(f"{fake_prob:.1f}%")
        
        st.markdown("🟢 Real News")
        st.progress(real_prob/100)
        st.caption(f"{real_prob:.1f}%")

st.markdown("---")
st.caption("Built with BERT + Streamlit | AI/ML Internship Project")