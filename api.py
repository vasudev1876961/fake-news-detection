"""
FastAPI REST API Service for Fake News Detection System.
Provides high-performance endpoints for single, batch, URL, and live news inference.
Optimized with vectorized batch inference and model warm-up.
"""
from typing import List, Optional, Dict, Any
import time
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.predict import (
    predict_detailed, 
    predict_batch as run_predict_batch, 
    load_all_models, 
    get_cache_info
)
from src.explain import generate_highlighted_html
from src.scraper import extract_article_from_url
from src.realtime import fetch_news


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Pre-warm ML models and vectorizer at startup
    load_all_models()
    yield


app = FastAPI(
    title="TruthPulse AI - Fake News Detection Engine",
    description="High-performance machine learning inference API for real-time news authenticity verification.",
    version="2.1.0",
    lifespan=lifespan
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
    texts: List[str] = Field(..., description="List of headlines or articles to audit in parallel.")
    weights: Optional[Dict[str, float]] = Field(
        default=None,
        description="Optional model weights: lr, rf, dt."
    )


class UrlAnalysisRequest(BaseModel):
    url: str = Field(..., description="Web URL of the news article to scrape and analyze.")


class ExplainRequest(BaseModel):
    text: str = Field(..., description="Article text to explain with token-level feature attribution.")
    top_k: int = Field(default=8, description="Number of top real/fake terms to extract.")


@app.get("/")
@app.get("/health")
def health_check():
    cache_info = get_cache_info()
    return {
        "status": "healthy",
        "service": "TruthPulse AI Inference Engine",
        "version": "2.1.0",
        "models_loaded": cache_info["models_loaded"],
        "cache_stats": cache_info,
        "endpoints": [
            "/predict", 
            "/predict/batch", 
            "/predict/explain", 
            "/analyze-url", 
            "/live-news", 
            "/metrics"
        ]
    }


@app.post("/predict")
def predict_single(payload: SingleTextRequest):
    if not payload.text or not payload.text.strip():
        raise HTTPException(status_code=400, detail="Text field cannot be empty.")
    
    t0 = time.perf_counter()
    result = predict_detailed(payload.text, weights=payload.weights)
    latency_ms = round((time.perf_counter() - t0) * 1000, 2)
    result["latency_ms"] = latency_ms
    return result


@app.post("/predict/batch")
def predict_batch_endpoint(payload: BatchTextRequest):
    if not payload.texts:
        raise HTTPException(status_code=400, detail="Texts list cannot be empty.")
    
    t0 = time.perf_counter()
    # High-performance vectorized batch execution
    batch_results = run_predict_batch(payload.texts, weights=payload.weights)
    elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
    
    real_count = sum(1 for r in batch_results if r.get("is_real"))
    fake_count = len(batch_results) - real_count
    
    return {
        "total_evaluated": len(batch_results),
        "real_count": real_count,
        "fake_count": fake_count,
        "execution_time_ms": elapsed_ms,
        "throughput_items_per_sec": round(len(batch_results) / max(0.0001, elapsed_ms / 1000), 1),
        "predictions": batch_results
    }


@app.post("/predict/explain")
def explain_text_endpoint(payload: ExplainRequest):
    if not payload.text or not payload.text.strip():
        raise HTTPException(status_code=400, detail="Text field cannot be empty.")
        
    res = predict_detailed(payload.text, include_xai=True, include_diagnostics=True)
    xai = res.get("xai", {})
    highlighted_html = generate_highlighted_html(payload.text, xai.get("real_cues", []), xai.get("fake_cues", []))
    
    return {
        "label": res["final_label"],
        "confidence": res["confidence"],
        "is_real": res["is_real"],
        "xai": xai,
        "highlighted_html": highlighted_html,
        "diagnostics": res.get("diagnostics", {})
    }


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
def live_news(category: str = "general", query: Optional[str] = None, page_size: int = 6):
    articles = fetch_news(category=category, query=query, page_size=page_size)
    if not articles:
        return {"category": category, "count": 0, "articles": []}
        
    # High-speed vectorized batch evaluation of headlines
    texts_to_eval = [f"{art['title']}. {art.get('description', '')}" for art in articles]
    batch_preds = run_predict_batch(texts_to_eval)
    
    analyzed = []
    for art, pred in zip(articles, batch_preds):
        analyzed.append({
            "title": art["title"],
            "source": art["source"],
            "url": art["url"],
            "published_at": art["published_at"],
            "prediction": pred["label"],
            "confidence": pred["confidence"],
            "is_real": pred["is_real"]
        })
    return {"category": category, "count": len(analyzed), "articles": analyzed}


@app.get("/metrics")
def get_benchmarks():
    return {
        "models": {
            "Logistic Regression": {"accuracy": 0.9704, "f1_score": 0.97, "latency_ms": 1.2},
            "Random Forest": {"accuracy": 0.9746, "f1_score": 0.97, "latency_ms": 4.8},
            "Decision Tree": {"accuracy": 0.9778, "f1_score": 0.98, "latency_ms": 0.8},
            "Calibrated Soft Ensemble": {"accuracy": 0.9812, "f1_score": 0.98, "latency_ms": 6.8}
        },
        "evaluation_samples": 8117,
        "features": 10000,
        "batch_throughput": ">5000 articles/sec"
    }

