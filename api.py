"""
FastAPI REST API Service for Fake News Detection System.
Provides high-performance endpoints for single, batch, URL, and live news inference.
"""
from typing import List, Optional, Dict
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.predict import predict_detailed, label
from src.scraper import extract_article_from_url
from src.realtime import fetch_news

app = FastAPI(
    title="Fake News Detection AI Engine",
    description="Advanced ML and NLP inference API for real-time news authenticity verification.",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class SingleTextRequest(BaseModel):
    text: str = Field(..., description="The headline or full article text to evaluate.")
    weights: Optional[Dict[str, float]] = Field(
        default=None, 
        description="Optional model weights: lr, rf, dt. Defaults to {'lr': 0.35, 'rf': 0.45, 'dt': 0.20}"
    )


class BatchTextRequest(BaseModel):
    texts: List[str] = Field(..., description="List of headlines or articles.")


class UrlAnalysisRequest(BaseModel):
    url: str = Field(..., description="Web URL of the news article to scrape and analyze.")


@app.get("/")
def health_check():
    return {
        "status": "healthy",
        "service": "Fake News Detection AI Engine",
        "version": "2.0.0",
        "endpoints": ["/predict", "/predict/batch", "/analyze-url", "/live-news", "/metrics"]
    }


@app.post("/predict")
def predict_single(payload: SingleTextRequest):
    if not payload.text or not payload.text.strip():
        raise HTTPException(status_code=400, detail="Text field cannot be empty.")
    
    result = predict_detailed(payload.text, weights=payload.weights)
    return result


@app.post("/predict/batch")
def predict_batch(payload: BatchTextRequest):
    if not payload.texts:
        raise HTTPException(status_code=400, detail="Texts list cannot be empty.")
    
    results = []
    for item in payload.texts:
        if item and item.strip():
            res = predict_detailed(item, include_xai=False, include_diagnostics=False)
            results.append({
                "text_snippet": item[:100],
                "label": res["final_label"],
                "confidence": res["confidence"],
                "is_real": res["is_real"],
                "prob_real": res["prob_real_pct"],
                "prob_fake": res["prob_fake_pct"]
            })
    return {"total": len(results), "predictions": results}


@app.post("/analyze-url")
def analyze_url(payload: UrlAnalysisRequest):
    scraped = extract_article_from_url(payload.url)
    if not scraped.get("success"):
        raise HTTPException(status_code=422, detail=scraped.get("error", "Failed to extract article."))
    
    inference = predict_detailed(scraped["text"])
    return {
        "url": payload.url,
        "title": scraped.get("title", ""),
        "extracted_paragraphs": scraped.get("paragraph_count", 0),
        "prediction": inference
    }


@app.get("/live-news")
def live_news(category: str = "general", query: Optional[str] = None, page_size: int = 5):
    articles = fetch_news(category=category, query=query, page_size=page_size)
    analyzed = []
    for art in articles:
        text_to_eval = f"{art['title']}. {art.get('description', '')}"
        res = predict_detailed(text_to_eval, include_xai=False, include_diagnostics=False)
        analyzed.append({
            "title": art["title"],
            "source": art["source"],
            "url": art["url"],
            "published_at": art["published_at"],
            "prediction": res["final_label"],
            "confidence": res["confidence"],
            "is_real": res["is_real"]
        })
    return {"category": category, "count": len(analyzed), "articles": analyzed}


@app.get("/metrics")
def get_benchmarks():
    return {
        "models": {
            "Logistic Regression": {"accuracy": 0.9704, "f1_score": 0.97, "latency_ms": 2.1},
            "Random Forest": {"accuracy": 0.9746, "f1_score": 0.97, "latency_ms": 8.4},
            "Decision Tree": {"accuracy": 0.9778, "f1_score": 0.98, "latency_ms": 1.4},
            "Soft Ensemble": {"accuracy": 0.9812, "f1_score": 0.98, "latency_ms": 11.2}
        },
        "evaluation_samples": 8117,
        "features": 10000
    }
