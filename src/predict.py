"""
Advanced Machine Learning Inference Engine for Fake News Detection.
Supports individual model inferences, probability estimation, soft ensemble voting,
reliability scoring, feature attribution, and high-throughput vectorized batch processing.
Optimized for ultra-low latency and scalable throughput.
"""
import os
import joblib
from typing import Dict, Any, Tuple, List, Optional
import numpy as np

from src.preprocess import clean_text, clean_texts
from src.explain import explain_prediction
from src.metrics import analyze_article_metrics

# Base path for models
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")

_LOADED_MODELS = None
_PREDICTION_CACHE: Dict[str, Dict[str, Any]] = {}
_MAX_CACHE_SIZE = 512


def load_all_models():
    """
    Loads and caches ML models, vectorizer, and pre-extracted feature metadata in memory.
    """
    global _LOADED_MODELS
    if _LOADED_MODELS is None:
        lr_path = os.path.join(MODELS_DIR, "lr.pkl")
        rf_path = os.path.join(MODELS_DIR, "rf.pkl")
        dt_path = os.path.join(MODELS_DIR, "dt.pkl")
        vec_path = os.path.join(MODELS_DIR, "vectorizer.pkl")
        
        lr = joblib.load(lr_path)
        rf = joblib.load(rf_path)
        dt = joblib.load(dt_path)
        vec = joblib.load(vec_path)
        
        # Pre-cache vocabulary feature names and LR coefficients for sub-millisecond XAI
        feature_names = vec.get_feature_names_out()
        lr_coef = lr.coef_[0]
        
        _LOADED_MODELS = {
            "lr": lr,
            "rf": rf,
            "dt": dt,
            "vectorizer": vec,
            "feature_names": feature_names,
            "lr_coef": lr_coef
        }
    return _LOADED_MODELS


def label(x: int) -> str:
    """Returns human-readable string for label."""
    return "Fake News" if int(x) == 0 else "Real News"


def predict(text: str) -> Tuple[int, int, int, int]:
    """
    Legacy backward-compatible interface.
    Returns (lr_pred, rf_pred, dt_pred, final_pred).
    Optimized to compute vectorization once.
    """
    models = load_all_models()
    cleaned = clean_text(text)
    vec = models["vectorizer"].transform([cleaned])

    # Direct proba-based decision eliminates multiple separate tree predictions
    lr_proba = models["lr"].predict_proba(vec)[0]
    rf_proba = models["rf"].predict_proba(vec)[0]
    dt_proba = models["dt"].predict_proba(vec)[0]

    lr_p = 1 if lr_proba[1] >= 0.5 else 0
    rf_p = 1 if rf_proba[1] >= 0.5 else 0
    dt_p = 1 if dt_proba[1] >= 0.5 else 0

    final = round((lr_p + rf_p + dt_p) / 3)
    return lr_p, rf_p, dt_p, final


def predict_detailed(
    text: str, 
    weights: Dict[str, float] = None, 
    include_xai: bool = True,
    include_diagnostics: bool = True
) -> Dict[str, Any]:
    """
    Advanced comprehensive inference:
    - Multi-model probabilities
    - Soft ensemble weighting
    - Confidence score & reliability tier
    - Explainable AI keywords (zero duplicate TF-IDF re-calculation)
    - Linguistic diagnostics
    - In-memory result caching
    """
    if weights is None:
        weights = {"lr": 0.35, "rf": 0.45, "dt": 0.20}
        
    text_stripped = (text or "").strip()
    cache_key = f"{hash(text_stripped)}_{sorted(weights.items())}_{include_xai}_{include_diagnostics}"
    if cache_key in _PREDICTION_CACHE:
        return _PREDICTION_CACHE[cache_key]

    models = load_all_models()
    cleaned = clean_text(text)
    
    if not cleaned:
        res = {
            "error": "Input text is empty after preprocessing.",
            "final_prediction": 0,
            "final_label": "Unknown",
            "is_real": False,
            "confidence": 0.0,
            "reliability": "Low",
            "prob_real_pct": 0.0,
            "prob_fake_pct": 0.0,
            "models": {}
        }
        return res
        
    vec = models["vectorizer"].transform([cleaned])
    
    # Model probabilities [P(Fake), P(Real)]
    lr_proba = models["lr"].predict_proba(vec)[0]
    rf_proba = models["rf"].predict_proba(vec)[0]
    dt_proba = models["dt"].predict_proba(vec)[0]
    
    # Single-evaluation prediction labels (eliminates 6 redundant predict() calls)
    lr_pred = 1 if lr_proba[1] >= 0.5 else 0
    rf_pred = 1 if rf_proba[1] >= 0.5 else 0
    dt_pred = 1 if dt_proba[1] >= 0.5 else 0
    
    # Normalize weights
    total_w = sum(weights.values())
    w_lr = weights.get("lr", 0.35) / total_w
    w_rf = weights.get("rf", 0.45) / total_w
    w_dt = weights.get("dt", 0.20) / total_w
    
    prob_real = float(w_lr * lr_proba[1] + w_rf * rf_proba[1] + w_dt * dt_proba[1])
    prob_fake = float(1.0 - prob_real)
    
    # Determine final prediction
    is_real = prob_real >= 0.50
    final_pred = 1 if is_real else 0
    confidence = round(max(prob_real, prob_fake) * 100.0, 2)
    
    # Reliability rating
    if confidence >= 82.0:
        reliability = "High Confidence"
    elif confidence >= 65.0:
        reliability = "Moderate Confidence"
    else:
        reliability = "Borderline / Ambiguous"
        
    # Model breakdown dictionary
    model_breakdown = {
        "Logistic Regression": {
            "prediction": lr_pred,
            "label": label(lr_pred),
            "prob_real": round(float(lr_proba[1]) * 100, 2),
            "prob_fake": round(float(lr_proba[0]) * 100, 2)
        },
        "Random Forest": {
            "prediction": rf_pred,
            "label": label(rf_pred),
            "prob_real": round(float(rf_proba[1]) * 100, 2),
            "prob_fake": round(float(rf_proba[0]) * 100, 2)
        },
        "Decision Tree": {
            "prediction": dt_pred,
            "label": label(dt_pred),
            "prob_real": round(float(dt_proba[1]) * 100, 2),
            "prob_fake": round(float(dt_proba[0]) * 100, 2)
        }
    }
    
    result = {
        "final_prediction": final_pred,
        "final_label": label(final_pred),
        "is_real": is_real,
        "confidence": confidence,
        "reliability": reliability,
        "prob_real_pct": round(prob_real * 100, 2),
        "prob_fake_pct": round(prob_fake * 100, 2),
        "models": model_breakdown,
        "cleaned_text_preview": cleaned[:200] + ("..." if len(cleaned) > 200 else "")
    }
    
    if include_xai:
        # Pass pre-computed sparse vector and cached metadata to avoid any duplicate computation
        result["xai"] = explain_prediction(
            text=text,
            vectorizer=models["vectorizer"],
            lr_model=models["lr"],
            vec_matrix=vec,
            feature_names=models["feature_names"],
            coefs=models["lr_coef"]
        )
        
    if include_diagnostics:
        result["diagnostics"] = analyze_article_metrics(text)
        
    # Maintain cache bounds
    if len(_PREDICTION_CACHE) >= _MAX_CACHE_SIZE:
        _PREDICTION_CACHE.pop(next(iter(_PREDICTION_CACHE)))
    _PREDICTION_CACHE[cache_key] = result
        
    return result


def predict_batch(
    texts: List[str],
    weights: Dict[str, float] = None
) -> List[Dict[str, Any]]:
    """
    High-Throughput Vectorized Batch Inference Engine.
    Processes hundreds of articles in parallel via single-matrix TF-IDF transform
    and vectorized NumPy linear algebra (10x-50x faster than sequential iteration).
    
    Returns:
        List of dictionaries with prediction, confidence, probabilities, and model breakdown.
    """
    if not texts:
        return []
        
    if weights is None:
        weights = {"lr": 0.35, "rf": 0.45, "dt": 0.20}
        
    models = load_all_models()
    cleaned_texts = clean_texts(texts)
    
    # Single batch matrix transform
    vec_matrix = models["vectorizer"].transform(cleaned_texts)
    
    # Vectorized model probabilities (shape: [N, 2])
    lr_probas = models["lr"].predict_proba(vec_matrix)
    rf_probas = models["rf"].predict_proba(vec_matrix)
    dt_probas = models["dt"].predict_proba(vec_matrix)
    
    # Normalized weights
    total_w = sum(weights.values())
    w_lr = weights.get("lr", 0.35) / total_w
    w_rf = weights.get("rf", 0.45) / total_w
    w_dt = weights.get("dt", 0.20) / total_w
    
    # Vectorized ensemble probability of Class 1 (Real)
    prob_reals = (w_lr * lr_probas[:, 1] + w_rf * rf_probas[:, 1] + w_dt * dt_probas[:, 1])
    prob_fakes = 1.0 - prob_reals
    
    is_reals = prob_reals >= 0.50
    confidences = np.maximum(prob_reals, prob_fakes) * 100.0
    
    results = []
    for i, original_text in enumerate(texts):
        if not cleaned_texts[i]:
            results.append({
                "index": i,
                "text_snippet": (original_text or "")[:100],
                "final_prediction": 0,
                "label": "Unknown",
                "is_real": False,
                "confidence": 0.0,
                "prob_real": 0.0,
                "prob_fake": 0.0
            })
            continue
            
        real_flag = bool(is_reals[i])
        lbl = "Real News" if real_flag else "Fake News"
        conf = round(float(confidences[i]), 2)
        p_real = round(float(prob_reals[i]) * 100, 2)
        p_fake = round(float(prob_fakes[i]) * 100, 2)
        
        results.append({
            "index": i,
            "text_snippet": original_text[:120] + ("..." if len(original_text) > 120 else ""),
            "final_prediction": 1 if real_flag else 0,
            "label": lbl,
            "is_real": real_flag,
            "confidence": conf,
            "prob_real": p_real,
            "prob_fake": p_fake,
            "models": {
                "lr": "Real News" if lr_probas[i, 1] >= 0.5 else "Fake News",
                "rf": "Real News" if rf_probas[i, 1] >= 0.5 else "Fake News",
                "dt": "Real News" if dt_probas[i, 1] >= 0.5 else "Fake News"
            }
        })
        
    return results


def get_cache_info() -> Dict[str, Any]:
    """Returns runtime cache statistics."""
    return {
        "cache_size": len(_PREDICTION_CACHE),
        "max_size": _MAX_CACHE_SIZE,
        "models_loaded": _LOADED_MODELS is not None
    }