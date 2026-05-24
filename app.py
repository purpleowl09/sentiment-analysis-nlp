import streamlit as st
import torch
import pandas as pd
import numpy as np
import re
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk

nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

# ── Page config ──────────────────────────────────────────
st.set_page_config(page_title="Movie Review Analyser", page_icon="🎬", layout="centered")
st.title("🎬 Movie Review Analyser")
st.markdown("Analyses sentiment and discovers topics using DistilBERT + LDA")

# ── Load models ──────────────────────────────────────────
@st.cache_resource
def load_models():
    # Sentiment model
    tokenizer = AutoTokenizer.from_pretrained("model/distilbert-sentiment")
    model = AutoModelForSequenceClassification.from_pretrained("model/distilbert-sentiment")
    model.eval()

    # Topic model
    df = pd.read_csv("train.tsv", delimiter='\t', header=None, names=['text', 'label'])
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    stop_words.update(['lrb', 'rrb', 'film', 'movie', 'one', 'make'])

    def clean(text):
        text = text.lower()
        text = re.sub(r'[^a-z\s]', '', text)
        words = text.split()
        words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
        return ' '.join(words)

    df['clean_text'] = df['text'].apply(clean)
    vectorizer = CountVectorizer(max_features=2000, min_df=5, max_df=0.9)
    X = vectorizer.fit_transform(df['clean_text'])
    lda = LatentDirichletAllocation(n_components=8, random_state=42, max_iter=20)
    lda.fit(X)

    return tokenizer, model, vectorizer, lda, lemmatizer, stop_words

with st.spinner("Loading models..."):
    tokenizer, model, vectorizer, lda, lemmatizer, stop_words = load_models()

st.success("Models loaded!")

# ── Topic names ───────────────────────────────────────────
topic_names = {
    0: "Comedy/Character", 1: "General Fun", 2: "Bad Comedies", 3: "Drama/Romance",
    4: "Performance Drama", 5: "Director/Craft", 6: "Feel-good Stories", 7: "Comedy Performance"
}

# ── Input ─────────────────────────────────────────────────
st.markdown("---")
user_input = st.text_area("Enter a movie review:", height=150,
    placeholder="e.g. This film was a masterpiece — beautifully acted and deeply moving.")

if st.button("Analyse", type="primary"):
    if not user_input.strip():
        st.warning("Please enter a review first.")
    else:
        # ── Sentiment ─────────────────────────────────────
        inputs = tokenizer(user_input, return_tensors="pt", truncation=True,
                           padding=True, max_length=64)
        with torch.no_grad():
            outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)[0]
        label = "Positive 😊" if probs[1] > probs[0] else "Negative 😞"
        confidence = max(probs).item()

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Sentiment", label)
        with col2:
            st.metric("Confidence", f"{confidence:.1%}")

        st.progress(float(probs[1]))
        st.caption(f"Positive probability: {probs[1]:.1%} | Negative: {probs[0]:.1%}")

        # ── Topic ─────────────────────────────────────────
        st.markdown("---")
        st.subheader("Topic Analysis")

        def clean(text):
            text = text.lower()
            text = re.sub(r'[^a-z\s]', '', text)
            words = text.split()
            words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
            return ' '.join(words)

        cleaned = clean(user_input)
        vec = vectorizer.transform([cleaned])
        topic_dist = lda.transform(vec)[0]

        top3_idx = topic_dist.argsort()[-3:][::-1]
        for idx in top3_idx:
            st.write(f"**{topic_names[idx]}** — {topic_dist[idx]:.1%}")
            st.progress(float(topic_dist[idx]))

# ── Footer ────────────────────────────────────────────────
st.markdown("---")
st.caption("Built with DistilBERT + LDA | NLP project by Keerthi")