# 🎬 Movie Review Sentiment Analyser

An end-to-end NLP project combining sentiment analysis and topic modelling on movie reviews.

🔗 **[Live Demo](https://sentiment-analysis-nlp01.streamlit.app/)** | 🤗 **[Model on Hugging Face](https://huggingface.co/Kee09/distilbert-sentiment-sst2)** | ⭐ **[GitHub](https://github.com/purpleowl09/sentiment-analysis-nlp)**

## What it does
- Classifies movie reviews as positive or negative using fine-tuned **DistilBERT** (99.8% confidence)
- Discovers hidden themes using **LDA topic modelling** — no labels needed
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
- **Model hosting:** Hugging Face Hub

## Project structure
day1.ipynb  → Environment setup + data loading
day2.ipynb  → Exploratory data analysis (5 charts)
day3.ipynb  → Text preprocessing + baseline models
day4.ipynb  → DistilBERT fine-tuning (Apple MPS GPU)
day5.ipynb  → LDA topic modelling
app.py      → Streamlit web app
## Key findings
- Fine-tuned DistilBERT achieves 88.9% accuracy vs 79.2% TF-IDF baseline (+9.7%)
- Trained 67M parameter model on Apple MPS GPU in under 4 minutes
- Drama/Romance reviews are most positive (65.1%)
- Bad Comedy reviews are least positive (34.3%)
- Model published on Hugging Face Hub: Kee09/distilbert-sentiment-sst2

## Run locally
```bash
git clone https://github.com/purpleowl09/sentiment-analysis-nlp
cd sentiment-analysis-nlp
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Author
Keerthi Girish — NLP + Data Science internship portfolio project  
🔗 [GitHub](https://github.com/purpleowl09) | 🤗 [Hugging Face](https://huggingface.co/Kee09)