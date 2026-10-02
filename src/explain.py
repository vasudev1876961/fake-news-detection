"""
Explainable AI (XAI) module for Fake News Detection.
Uses Logistic Regression feature coefficients and TF-IDF term weights to extract
the strongest influential tokens driving Real vs Fake classification.
Optimized with vectorized NumPy sparse operations and token highlighting.
"""
from typing import Dict, List, Any, Optional
import html
import re
import numpy as np


def explain_prediction(
    text: str, 
    vectorizer, 
    lr_model, 
    top_k: int = 6,
    vec_matrix: Optional[Any] = None,
    feature_names: Optional[np.ndarray] = None,
    coefs: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    """
    Extracts top terms in the given text that push towards Fake or Real news.
    Optimized with vectorized NumPy sparse array operations.
    
    Returns:
        dict with:
            - real_cues: list of {term, score}
            - fake_cues: list of {term, score}
            - evidence_score: float
            - total_matched_features: int
    """
    if not text or vectorizer is None or lr_model is None:
        return {"real_cues": [], "fake_cues": [], "evidence_score": 0.0, "total_matched_features": 0}
    
    try:
        # 1. Reuse or compute sparse TF-IDF matrix
        if vec_matrix is None:
            from src.preprocess import clean_text
            cleaned = clean_text(text)
            if not cleaned:
                return {"real_cues": [], "fake_cues": [], "evidence_score": 0.0, "total_matched_features": 0}
            vec_matrix = vectorizer.transform([cleaned])
            
        # 2. Reuse or cache feature names and coefficients
        if feature_names is None:
            if not hasattr(vectorizer, "_cached_feature_names"):
                vectorizer._cached_feature_names = vectorizer.get_feature_names_out()
            feature_names = vectorizer._cached_feature_names
            
        if coefs is None:
            if not hasattr(lr_model, "_cached_coefs"):
                lr_model._cached_coefs = lr_model.coef_[0]
            coefs = lr_model._cached_coefs
            
        # Direct C-level CSR indices & data (avoids nonzero() overhead)
        nonzero_indices = vec_matrix.indices
        nonzero_data = vec_matrix.data
        
        if len(nonzero_indices) == 0:
            return {"real_cues": [], "fake_cues": [], "evidence_score": 0.0, "total_matched_features": 0}
            
        # Vectorized weights: TF-IDF * feature coefficient
        term_coefs = coefs[nonzero_indices]
        weights = nonzero_data * term_coefs
        matched_terms = feature_names[nonzero_indices]
        
        # Split positive (Real cues) and negative (Fake cues)
        pos_mask = weights > 0
        neg_mask = weights < 0
        
        real_cues = []
        if np.any(pos_mask):
            pos_indices = np.argsort(weights[pos_mask])[::-1][:top_k]
            pos_terms = matched_terms[pos_mask][pos_indices]
            pos_scores = weights[pos_mask][pos_indices]
            real_cues = [{"term": str(t), "score": round(float(s), 3)} for t, s in zip(pos_terms, pos_scores)]
            
        fake_cues = []
        if np.any(neg_mask):
            neg_indices = np.argsort(weights[neg_mask])[:top_k]
            neg_terms = matched_terms[neg_mask][neg_indices]
            neg_scores = np.abs(weights[neg_mask][neg_indices])
            fake_cues = [{"term": str(t), "score": round(float(s), 3)} for t, s in zip(neg_terms, neg_scores)]
            
        total_evidence = float(np.sum(weights))
        
        return {
            "real_cues": real_cues,
            "fake_cues": fake_cues,
            "evidence_score": round(total_evidence, 3),
            "total_matched_features": int(len(nonzero_indices))
        }
    except Exception as e:
        return {
            "real_cues": [],
            "fake_cues": [],
            "evidence_score": 0.0,
            "total_matched_features": 0,
            "error": str(e)
        }


def generate_highlighted_html(text: str, real_cues: List[Dict[str, Any]], fake_cues: List[Dict[str, Any]]) -> str:
    """
    Renders article text with color-coded highlighting:
    - Emerald Green for verified journalistic tokens
    - Rose/Crimson for sensationalist/clickbait tokens
    """
    if not text:
        return ""
        
    escaped_text = html.escape(text)
    
    # Sort terms by length descending to match multi-word n-grams first
    real_terms = sorted([c["term"] for c in real_cues if c.get("term")], key=len, reverse=True)
    fake_terms = sorted([c["term"] for c in fake_cues if c.get("term")], key=len, reverse=True)
    
    for term in real_terms:
        pattern = re.compile(rf'\b({re.escape(term)})\b', re.IGNORECASE)
        escaped_text = pattern.sub(
            r'<span style="background-color: rgba(16, 185, 129, 0.25); border-bottom: 2px solid #10B981; padding: 1px 4px; border-radius: 4px; font-weight: 600; color: #34D399;" title="Journalistic pattern (Real cue)">\1</span>',
            escaped_text
        )
        
    for term in fake_terms:
        pattern = re.compile(rf'\b({re.escape(term)})\b', re.IGNORECASE)
        escaped_text = pattern.sub(
            r'<span style="background-color: rgba(244, 63, 94, 0.25); border-bottom: 2px solid #F43F5E; padding: 1px 4px; border-radius: 4px; font-weight: 600; color: #FB7185;" title="Sensational/unverified cue">\1</span>',
            escaped_text
        )
        
    return f'<div style="line-height: 1.8; font-size: 0.98rem; color: #E2E8F0; padding: 12px; background: rgba(15, 23, 42, 0.6); border-radius: 8px; border: 1px solid rgba(255,255,255,0.08); max-height: 380px; overflow-y: auto;">{escaped_text}</div>'

