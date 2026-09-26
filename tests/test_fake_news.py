"""
Comprehensive Unit Tests for Fake News Detection System.
"""
import pytest
from src.preprocess import clean_text, get_text_statistics
from src.metrics import calculate_sensationalism_index, calculate_lexical_diversity
from src.predict import predict, predict_detailed, label
from src.scraper import extract_article_from_url


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


def test_text_statistics():
    sample = "This is the first sentence. Here is another sentence with more words."
    stats = get_text_statistics(sample)
    assert stats["word_count"] > 5
    assert stats["sentence_count"] == 2
    assert stats["avg_word_length"] > 0


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


def test_scraper_invalid_url():
    res = extract_article_from_url("http://invalid-domain-that-does-not-exist-12345xyz.com")
    assert res["success"] is False
    assert "error" in res
