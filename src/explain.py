"""
Explainable AI (XAI) module for Fake News Detection.
Uses Logistic Regression feature coefficients and TF-IDF term weights to extract
the strongest influential tokens driving Real vs Fake classification.
"""
from typing import Dict, List, Any
import numpy as np


def explain_prediction(text: str, vectorizer, lr_model, top_k: int = 6) -> Dict[str, Any]:
    """
    Extracts top terms in the given text that push towards Fake or Real news.
    
    Returns:
        dict with:
            - real_cues: list of {term, score}
            - fake_cues: list of {term, score}
            - feature_evidence: float
    """
    if not text or not vectorizer or not lr_model:
        return {"real_cues": [], "fake_cues": [], "evidence_score": 0.0}
    
    try:
        from src.preprocess import clean_text
        cleaned = clean_text(text)
        if not cleaned:
            return {"real_cues": [], "fake_cues": [], "evidence_score": 0.0}
            
        vec_matrix = vectorizer.transform([cleaned])
        feature_names = vectorizer.get_feature_names_out()
        coefs = lr_model.coef_[0]
        
        # Non-zero indices in sparse vector
        nonzero_indices = vec_matrix.nonzero()[1]
        
        contributions = []
        for idx in nonzero_indices:
            term = feature_names[idx]
            tfidf_val = vec_matrix[0, idx]
            weight = coefs[idx] * tfidf_val
            contributions.append((term, float(weight)))
            
        # Sort by weight
        # Positive weights push toward Class 1 (Real)
        # Negative weights push toward Class 0 (Fake)
        contributions.sort(key=lambda x: x[1], reverse=True)
        
        real_cues = [
            {"term": term, "score": round(score, 3)}
            for term, score in contributions if score > 0
        ][:top_k]
        
        fake_cues = [
            {"term": term, "score": round(abs(score), 3)}
            for term, score in reversed(contributions) if score < 0
        ][:top_k]
        
        total_evidence = sum(c[1] for c in contributions)
        
        return {
            "real_cues": real_cues,
            "fake_cues": fake_cues,
            "evidence_score": round(total_evidence, 3),
            "total_matched_features": len(nonzero_indices)
        }
    except Exception as e:
        return {
            "real_cues": [],
            "fake_cues": [],
            "evidence_score": 0.0,
            "error": str(e)
        }
