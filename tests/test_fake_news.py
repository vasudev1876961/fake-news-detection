"""
Comprehensive Unit & Integration Tests for Fake News Detection System.
Covers preprocessing, stylistic metrics, model inferences (single and batch),
explainable AI, caching, and FastAPI endpoints.
"""
import pytest
from fastapi.testclient import TestClient

from src.preprocess import clean_text, clean_texts, get_text_statistics
from src.metrics import calculate_sensationalism_index, calculate_lexical_diversity
from src.predict import predict, predict_detailed, predict_batch, label, get_cache_info
from src.explain import generate_highlighted_html
from src.scraper import extract_article_from_url
from api import app


def test_clean_text_basic():
    raw = "BREAKING: Check this out at https://example.com [Reuters] 12345!"
    cleaned = clean_text(raw)
    assert "https" not in cleaned
    assert "reuters" not in cleaned
    assert "12345" not in cleaned
    assert "breaking check this out" in cleaned


def test_clean_text_empty():
    assert clean_text(None) == ""
    assert clean_text("") == ""


def test_clean_texts_batch():
    raw_list = ["Item one [link]", "Item two https://test.org 456"]
    cleaned_list = clean_texts(raw_list)
    assert len(cleaned_list) == 2
    assert "link" not in cleaned_list[0]
    assert "456" not in cleaned_list[1]


def test_text_statistics():
    sample = "This is the first sentence. Here is another sentence with more words."
    stats = get_text_statistics(sample)
    assert stats["word_count"] > 5
    assert stats["sentence_count"] == 2
    assert stats["avg_word_length"] > 0
    assert "reading_ease" in stats
    assert "reading_level" in stats


def test_sensationalism_index():
    sensational_text = "SHOCKING BOMBSHELL EXPOSED! You will not believe this scandal!!!"
    res = calculate_sensationalism_index(sensational_text)
    assert res["score"] > 30.0
    assert "shocking" in res["triggers_found"] or "bombshell" in res["triggers_found"] or "scandal" in res["triggers_found"]

    calm_text = "The quarterly inflation report was released on Tuesday morning by the department."
    res_calm = calculate_sensationalism_index(calm_text)
    assert res_calm["score"] < res["score"]


def test_lexical_diversity():
    repetitive = "test test test test"
    diverse = "apple banana orange grape strawberry melon"
    assert calculate_lexical_diversity(diverse) > calculate_lexical_diversity(repetitive)


def test_model_predict_legacy():
    lr_p, rf_p, dt_p, final = predict("WASHINGTON (Reuters) - Congress passed the tax reform bill.")
    assert final in (0, 1)
    assert label(final) in ("Fake News", "Real News")


def test_model_predict_detailed():
    res = predict_detailed("WASHINGTON (Reuters) - The Senate approved the bipartisan legislation.")
    assert "final_label" in res
    assert "confidence" in res
    assert "models" in res
    assert "Logistic Regression" in res["models"]
    assert "Random Forest" in res["models"]
    assert "Decision Tree" in res["models"]
    assert res["confidence"] >= 50.0
    assert "xai" in res
    assert "diagnostics" in res


def test_model_predict_batch_vectorized():
    texts = [
        "WASHINGTON (Reuters) - Federal officials confirmed economic growth figures on Friday.",
        "SHOCKING CONSPIRACY: Secret alien clone found in bunker, doctors amazed!",
        "The committee convened to review agricultural tariffs."
    ]
    batch_res = predict_batch(texts)
    assert len(batch_res) == 3
    for item in batch_res:
        assert "label" in item
        assert "confidence" in item
        assert "is_real" in item
        assert item["confidence"] >= 50.0
    
    # First item should be real news
    assert batch_res[0]["is_real"] is True


def test_prediction_caching():
    text = "WASHINGTON (Reuters) - Routine treasury announcement."
    _ = predict_detailed(text)
    cache_stats = get_cache_info()
    assert cache_stats["cache_size"] > 0


def test_highlighted_html_generator():
    text = "WASHINGTON (Reuters) - SHOCKING news occurred today."
    real_cues = [{"term": "reuters", "score": 2.5}]
    fake_cues = [{"term": "shocking", "score": 1.8}]
    html_out = generate_highlighted_html(text, real_cues, fake_cues)
    assert "reuters" in html_out.lower()
    assert "#10b981" in html_out.lower() or "rgba(16, 185, 129" in html_out.lower()
    assert "#f43f5e" in html_out.lower() or "rgba(244, 63, 94" in html_out.lower()


def test_scraper_invalid_url():
    res = extract_article_from_url("http://invalid-domain-that-does-not-exist-12345xyz.com")
    assert res["success"] is False
    assert "error" in res


def test_fastapi_endpoints():
    client = TestClient(app)
    
    # 1. Health check
    h_res = client.get("/health")
    assert h_res.status_code == 200
    assert h_res.json()["status"] == "healthy"
    
    # 2. Single predict
    p_res = client.post("/predict", json={"text": "WASHINGTON (Reuters) - Leaders reached agreement."})
    assert p_res.status_code == 200
    assert p_res.json()["is_real"] is True
    assert "latency_ms" in p_res.json()
    
    # 3. Vectorized Batch predict
    b_res = client.post("/predict/batch", json={"texts": ["Article 1", "Article 2"]})
    assert b_res.status_code == 200
    assert b_res.json()["total_evaluated"] == 2
    assert "execution_time_ms" in b_res.json()
    
    # 4. Explain endpoint
    e_res = client.post("/predict/explain", json={"text": "Reuters reports economic growth."})
    assert e_res.status_code == 200
    assert "highlighted_html" in e_res.json()

