"""
Text linguistic diagnostics and content quality analytics:
- Sensationalism & Clickbait Index
- Lexical Diversity (Type-Token Ratio)
- Readability Assessment
- Subjectivity & Stylometry Markers
"""
import re
from typing import Dict, Any


SENSATIONAL_KEYWORDS = {
    "shocking", "unbelievable", "bombshell", "exposed", "conspiracy",
    "secret", "you won't believe", "mindblowing", "bizarre", "jaw-dropping",
    "destroy", "slams", "blast", "furious", "outrage", "traitor",
    "corrupt", "cover-up", "treason", "miracle", "hoax", "scandal",
    "hidden truth", "plot", "devastating", "panic", "insane", "viral"
}


def calculate_sensationalism_index(text: str) -> Dict[str, Any]:
    """
    Computes a sensationalism / clickbait score (0 - 100%) based on:
    - Capitalization shouting ratio
    - Exclamation mark density
    - Trigger/buzzword frequency
    """
    if not text:
        return {"score": 0.0, "level": "None", "triggers_found": []}
    
    text_str = str(text)
    words = re.findall(r'\b[A-Za-z]+\b', text_str)
    if not words:
        return {"score": 0.0, "level": "None", "triggers_found": []}
    
    total_words = len(words)
    
    # Capitalized shouting words (excluding single letters like 'I' or 'A')
    shout_words = [w for w in words if len(w) > 1 and w.isupper()]
    shout_ratio = min(1.0, len(shout_words) / max(1, total_words))
    
    # Exclamation mark density
    exclamation_count = text_str.count('!')
    exclamation_ratio = min(1.0, (exclamation_count * 2) / max(1, total_words))
    
    # Sensational keywords detection
    lower_text = text_str.lower()
    found_triggers = [kw for kw in SENSATIONAL_KEYWORDS if kw in lower_text]
    trigger_ratio = min(1.0, len(found_triggers) / 5.0)
    
    # Weighted composite score
    composite = (shout_ratio * 0.35 + exclamation_ratio * 0.25 + trigger_ratio * 0.40) * 100.0
    score = round(min(100.0, max(0.0, composite)), 1)
    
    if score < 25:
        level = "Low (Formal / Objective)"
    elif score < 60:
        level = "Moderate (Casual / Tabloid)"
    else:
        level = "High (Sensationalist / Clickbait)"
        
    return {
        "score": score,
        "level": level,
        "triggers_found": found_triggers[:8],
        "shout_words_count": len(shout_words),
        "exclamation_count": exclamation_count
    }


def calculate_lexical_diversity(text: str) -> float:
    """
    Computes Type-Token Ratio (unique words / total words).
    Higher ratio indicates richer, more varied vocabulary.
    """
    words = re.findall(r'\b[A-Za-z]+\b', str(text).lower())
    if not words:
        return 0.0
    return round((len(set(words)) / len(words)) * 100.0, 1)


def analyze_article_metrics(text: str) -> Dict[str, Any]:
    """
    Runs full suite of text diagnostics on an article.
    """
    from src.preprocess import get_text_statistics
    
    stats = get_text_statistics(text)
    sensationalism = calculate_sensationalism_index(text)
    lexical_diversity = calculate_lexical_diversity(text)
    
    return {
        "stats": stats,
        "sensationalism": sensationalism,
        "lexical_diversity_pct": lexical_diversity
    }
