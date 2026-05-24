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
day1.ipynb  → Environment setup + data loading
day2.ipynb  → Exploratory data analysis (5 charts)
day3.ipynb  → Text preprocessing + baseline models
day4.ipynb  → DistilBERT fine-tuning
day5.ipynb  → LDA topic modelling
app.py      → Streamlit web app
## Key findings
- DistilBERT achieves 88.9% accuracy vs 79.2% TF-IDF baseline
- Drama/Romance reviews are most positive (65.1%)
- Bad Comedy reviews are least positive (34.3%)
- Trained on Apple MPS GPU in under 4 minutes

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
Keerthi Girish — Built as an NLP internship portfolio project