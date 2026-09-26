"""
Advanced Machine Learning Inference Engine for Fake News Detection.
Supports individual model inferences, probability estimation, soft ensemble voting,
reliability scoring, and feature attribution.
"""
import os
import joblib
from typing import Dict, Any, Tuple
from src.preprocess import clean_text
from src.explain import explain_prediction
from src.metrics import analyze_article_metrics

# Base path for models
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")

_LOADED_MODELS = None


def load_all_models():
    """
    Loads and caches ML models and vectorizer in memory.
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
        
        _LOADED_MODELS = {
            "lr": lr,
            "rf": rf,
            "dt": dt,
            "vectorizer": vec
        }
    return _LOADED_MODELS


def label(x: int) -> str:
    """Returns human-readable string for label."""
    return "Fake News" if int(x) == 0 else "Real News"


def predict(text: str) -> Tuple[int, int, int, int]:
    """
    Legacy backward-compatible interface.
    Returns (lr_pred, rf_pred, dt_pred, final_pred).
    """
    models = load_all_models()
    cleaned = clean_text(text)
    vec = models["vectorizer"].transform([cleaned])

    lr_p = int(models["lr"].predict(vec)[0])
    rf_p = int(models["rf"].predict(vec)[0])
    dt_p = int(models["dt"].predict(vec)[0])

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
    - Explainable AI keywords
    - Linguistic diagnostics
    """
    if weights is None:
        weights = {"lr": 0.35, "rf": 0.45, "dt": 0.20}
        
    models = load_all_models()
    cleaned = clean_text(text)
    
    if not cleaned:
        return {
            "error": "Input text is empty after preprocessing.",
            "final_label": "Unknown",
            "is_real": False,
            "confidence": 0.0
        }
        
    vec = models["vectorizer"].transform([cleaned])
    
    # Model probabilities [P(Fake), P(Real)]
    lr_proba = models["lr"].predict_proba(vec)[0]
    rf_proba = models["rf"].predict_proba(vec)[0]
    dt_proba = models["dt"].predict_proba(vec)[0]
    
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
            "prediction": int(models["lr"].predict(vec)[0]),
            "label": label(models["lr"].predict(vec)[0]),
            "prob_real": round(float(lr_proba[1]) * 100, 2),
            "prob_fake": round(float(lr_proba[0]) * 100, 2)
        },
        "Random Forest": {
            "prediction": int(models["rf"].predict(vec)[0]),
            "label": label(models["rf"].predict(vec)[0]),
            "prob_real": round(float(rf_proba[1]) * 100, 2),
            "prob_fake": round(float(rf_proba[0]) * 100, 2)
        },
        "Decision Tree": {
            "prediction": int(models["dt"].predict(vec)[0]),
            "label": label(models["dt"].predict(vec)[0]),
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
        result["xai"] = explain_prediction(text, models["vectorizer"], models["lr"])
        
    if include_diagnostics:
        result["diagnostics"] = analyze_article_metrics(text)
        
    return result