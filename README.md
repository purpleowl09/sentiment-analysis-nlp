# 🎬 Movie Review Sentiment Analyser

An end-to-end NLP project combining sentiment analysis and topic modelling on movie reviews.

🔗 **[Live Demo](https://sentiment-analysis-nlp01.streamlit.app/)**

## What it does
- Classifies movie reviews as positive or negative using fine-tuned **DistilBERT**
- Discovers hidden themes using **LDA topic modelling**
- Shows confidence scores and top 3 relevant topics for any review

## Results

| Model | Accuracy |
|-------|----------|
| TF-IDF + Logistic Regression (baseline) | 79.2% |
| TF-IDF + Naive Bayes | 79.8% |
| **DistilBERT (fine-tuned)** | **88.9%** |

**+9.7% improvement over baseline**

## Tech stack
- **NLP:** HuggingFace Transformers, DistilBERT, NLTK
- **ML:** scikit-learn, PyTorch
- **Data:** Pandas, NumPy
- **Visualisation:** Matplotlib, Seaborn, WordCloud
- **App:** Streamlit

## Project structure