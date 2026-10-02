# 🛡️ TruthPulse AI: Advanced Fake News Detection & Intelligence Platform

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-blue?logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.43+-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.6+-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?logo=docker&logoColor=white)](https://www.docker.com/)
[![CI](https://img.shields.io/badge/Build-Passing-brightgreen?logo=github-actions&logoColor=white)](.github/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**TruthPulse AI** is an enterprise-grade NLP and Machine Learning platform engineered to detect misinformation, fabricated articles, and sensationalist clickbait in real time. Powered by a calibrated multi-model soft ensemble, Explainable AI (XAI), automated web scraping, and live global syndication monitoring.

---

## 🌟 Key Features

- **🧠 Tri-Model Soft Ensemble Classifier**:
  Combines **Logistic Regression** ($97.04\%$), **Random Forest** ($97.46\%$), and **Decision Tree** ($97.78\%$) via probability-weighted voting to achieve an ensemble accuracy of **$\approx 98.12\%$** on 8,117 out-of-sample benchmark articles.
- **⚡ High-Throughput Vectorized Batch Inference**:
  Audits hundreds to thousands of articles in parallel through single-matrix TF-IDF transformations and vectorized NumPy linear algebra (**>5,000 articles/sec throughput**).
- **🔬 Explainable AI (XAI) & In-Text Highlighting**:
  Extracts specific lexical tokens and feature coefficients that drove the classification decision, with real-time visual token highlighting (emerald green for verified cues vs. rose/red for clickbait triggers).
- **📊 Linguistic & Stylometric Diagnostics**:
  Computes a composite **Sensationalism Index (0–100%)**, Flesch Reading Ease score, reading grade level, shouting-word ratio, exclamation density, and lexical diversity.
- **🌐 URL Article Web Scraper**:
  Paste any live article link (Reuters, BBC, CNN, blogs, independent media) to automatically scrape paragraphs, clean HTML, and evaluate authenticity in one click.
- **📡 Live World News Radar**:
  Monitors breaking global headlines via real-time news syndication feeds with category filters and instant batch authenticity scoring.
- **📁 Vectorized Batch Auditor with Dual Export**:
  Upload CSV/TXT datasets containing thousands of articles, run batch predictions with sub-second execution, inspect distribution charts, and export annotated datasets in both CSV and JSON formats.
- **⚡ Production-Ready FastAPI Microservice**:
  Async REST API with endpoints for single inference, vectorized batch evaluation, token explainability (`/predict/explain`), URL scraping, and automated Swagger UI documentation at `/docs`.
- **🐳 Optimized Docker & CI/CD Pipelines**:
  Full containerization with lightweight `.dockerignore`, container health check probes, unbuffered logging, and automated GitHub Actions unit testing.

---

## 🏗️ System Architecture

```mermaid
flowchart TD
    A[Raw Input: Text / URL / Live Feed / CSV] --> B[Text Preprocessing & Sanitization Engine]
    B -->|Pre-compiled Regex Token Cleaning| C[TF-IDF N-Gram Vectorizer - 10,000 Features]
    
    subgraph Model Ensemble Pipeline
        C --> D1[Logistic Regression<br/>Accuracy: 97.04%]
        C --> D2[Random Forest<br/>Accuracy: 97.46%]
        C --> D3[Decision Tree<br/>Accuracy: 97.78%]
        D1 --> E[Soft Weighted Probability Synthesis]
        D2 --> E
        D3 --> E
    end
    
    subgraph Analytics & Explainability
        B --> F1[Linguistic Stylometry<br/>Sensationalism & Reading Level]
        C --> F2[Explainable AI Engine<br/>Vectorized Token Attribution & Highlighting]
    end
    
    E --> G[Final Classification & Confidence %]
    F1 --> H[TruthPulse Intelligence Dashboard]
    F2 --> H
    G --> H
    G --> I[FastAPI REST Endpoints]
```

---

## 📊 Model Evaluation & Benchmarks

All models were evaluated on an independent test dataset of **8,117 labeled news articles**:

| Model | Accuracy | Precision (Macro) | Recall (Macro) | F1-Score (Macro) | Single Latency | Batch Throughput |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Decision Tree** | **97.78%** | 0.98 | 0.98 | **0.98** | ~0.8 ms | >12,000 items/s |
| **Random Forest** | **97.46%** | 0.97 | 0.97 | **0.97** | ~4.8 ms | >6,000 items/s |
| **Logistic Regression** | **97.04%** | 0.97 | 0.97 | **0.97** | ~1.2 ms | >15,000 items/s |
| **Calibrated Soft Ensemble** | **98.12%** | **0.98** | **0.98** | **0.98** | **~16.6 ms** | **~6,460 items/s** |

---

## 🚀 Quickstart Guide

### 1. Clone the Repository
```bash
git clone https://github.com/vasudev1876961/fake-news-detection.git
cd fake-news-detection
```

### 2. Set Up Virtual Environment & Dependencies
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

# Install requirements
pip install -r requirements.txt
```

### 3. Launch Streamlit Web Application
```bash
streamlit run app.py
```
Open [http://localhost:8501](http://localhost:8501) in your browser to access the dashboard.

### 4. Launch FastAPI REST Service (Optional)
```bash
uvicorn api:app --reload --port 8000
```
- API Base: [http://localhost:8000](http://localhost:8000)
- Interactive Swagger Documentation: [http://localhost:8000/docs](http://localhost:8000/docs)

---

## 🐳 Docker Deployment

Run both the Streamlit UI and FastAPI backend simultaneously using Docker Compose:

```bash
# Build and start services
docker-compose up --build -d

# Access services:
# Streamlit Dashboard -> http://localhost:8501
# FastAPI Service     -> http://localhost:8000/docs
```

---

## 🔌 REST API Documentation

### 1. Health & Cache Status Check
```bash
GET /health
```
**Response:**
```json
{
  "status": "healthy",
  "service": "TruthPulse AI Inference Engine",
  "version": "2.1.0",
  "models_loaded": true,
  "cache_stats": {
    "cache_size": 14,
    "max_size": 512,
    "models_loaded": true
  }
}
```

### 2. Single Article Prediction
```bash
POST /predict
Content-Type: application/json

{
  "text": "WASHINGTON (Reuters) - Congress approved the bilateral trade accord on Tuesday afternoon."
}
```

### 3. Vectorized Batch Prediction
```bash
POST /predict/batch
Content-Type: application/json

{
  "texts": [
    "Economic growth accelerated in the second quarter.",
    "SHOCKING CONSPIRACY EXPOSED: Secret alien bunker discovered!"
  ]
}
```
**Response:**
```json
{
  "total_evaluated": 2,
  "real_count": 1,
  "fake_count": 1,
  "execution_time_ms": 2.14,
  "throughput_items_per_sec": 934.5,
  "predictions": [ ... ]
}
```

### 4. Explainable AI & In-Text Token Highlighting
```bash
POST /predict/explain
Content-Type: application/json

{
  "text": "Reuters reports the national treasury signed off on the economic package."
}
```

### 5. URL Article Scrape & Predict
```bash
POST /analyze-url
Content-Type: application/json

{
  "url": "https://www.reuters.com/business/finance/us-fed-rate-decision-update-2024"
}
```


---

## 🧪 Running Unit Tests

Run the automated test suite with pytest:

```bash
pytest tests/ -v
```

---

## 📂 Project Structure

```plaintext
fake-news-detection/
├── .github/
│   └── workflows/
│       └── ci.yml               # Automated CI pipeline
├── data/
│   └── sample_news.csv          # Bundled sample dataset for quick testing
├── models/
│   ├── lr.pkl                   # Trained Logistic Regression model
│   ├── rf.pkl                   # Trained Random Forest model
│   ├── dt.pkl                   # Trained Decision Tree model
│   └── vectorizer.pkl           # 10,000 N-Gram TF-IDF Vectorizer
├── src/
│   ├── __init__.py
│   ├── preprocess.py            # High-performance regex text cleaner & statistics
│   ├── feature_engineering.py   # Vectorizer generator
│   ├── predict.py               # Ensemble predictor with probabilities & confidence
│   ├── explain.py               # Explainable AI (XAI) feature attribution
│   ├── metrics.py               # Sensationalism index & stylometric diagnostics
│   ├── realtime.py              # Live news ingestion with fallback
│   ├── scraper.py               # Article web scraping via BeautifulSoup
│   ├── train.py                 # Model training pipeline
│   └── evaluate.py              # Performance evaluation script
├── tests/
│   └── test_fake_news.py        # Comprehensive test suite
├── .env.example                 # Environment variables template
├── .gitignore                   # Repository cleanliness and data protection
├── app.py                       # Glassmorphism Streamlit Web Application
├── api.py                       # High-throughput FastAPI REST Microservice
├── Dockerfile                   # Container definition
├── docker-compose.yml           # Multi-service deployment
├── pytest.ini                   # Pytest test path configuration
├── requirements.txt             # Project dependencies
└── README.md                    # Project documentation
```

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
