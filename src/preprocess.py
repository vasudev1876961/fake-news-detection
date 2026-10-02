import re
import string
from typing import List, Dict, Any

# Pre-compiled regular expressions for maximum preprocessing throughput
_RE_BRACKETS = re.compile(r'\[.*?\]')
_RE_URLS = re.compile(r'https?://\S+|www\.\S+')
_RE_HTML = re.compile(r'<.*?>+')
_RE_NEWLINES = re.compile(r'[\r\n]+')
_RE_DIGIT_WORDS = re.compile(r'\w*\d\w*')
_RE_WHITESPACE = re.compile(r'\s+')
_RE_WORDS = re.compile(r'\b[A-Za-z]+\b')
_RE_SENTENCES = re.compile(r'[.!?]+')

_PUNCT_TABLE = str.maketrans('', '', string.punctuation)


def clean_text(text: str) -> str:
    """
    Cleans raw text for TF-IDF vectorization and ML model inference.
    Handles HTML tags, URLs, brackets, punctuation, numbers, and multiple whitespaces.
    Optimized with pre-compiled regex engines for ultra-low latency.
    """
    if text is None:
        return ""
    
    text = str(text)
    if not text:
        return ""
        
    text = text.lower()
    
    # Strip bracketed citations/text [e.g., [Reuters]]
    text = _RE_BRACKETS.sub('', text)
    
    # Strip web URLs
    text = _RE_URLS.sub('', text)
    
    # Strip HTML tags
    text = _RE_HTML.sub('', text)
    
    # Strip newlines
    text = _RE_NEWLINES.sub(' ', text)
    
    # Strip words containing digits
    text = _RE_DIGIT_WORDS.sub('', text)
    
    # Strip punctuation
    text = text.translate(_PUNCT_TABLE)
    
    # Normalize whitespaces
    return _RE_WHITESPACE.sub(' ', text).strip()


def clean_texts(texts: List[str]) -> List[str]:
    """
    High-throughput batch cleaning function for collections of articles.
    """
    return [clean_text(t) for t in texts]


def get_text_statistics(text: str) -> Dict[str, Any]:
    """
    Computes key readability, structural, and grade-level metrics for text diagnostics.
    """
    if not text:
        return {
            "char_count": 0,
            "word_count": 0,
            "sentence_count": 0,
            "avg_word_length": 0.0,
            "reading_time_mins": 0.0,
            "reading_ease": 100.0,
            "reading_level": "Standard"
        }
    
    raw = str(text).strip()
    words = [w for w in _RE_WHITESPACE.split(raw) if w]
    sentences = [s for s in _RE_SENTENCES.split(raw) if s.strip()]
    
    word_count = len(words)
    char_count = len(raw)
    sentence_count = max(1, len(sentences))
    
    avg_word_length = round(sum(len(w) for w in words) / max(1, word_count), 2)
    # Average reading speed ~ 200 words per minute
    reading_time_mins = round(word_count / 200.0, 1)
    
    # Simplified Flesch Reading Ease approximation
    # 206.835 - 1.015 * (words/sentences) - 84.6 * (syllables/words)
    # Approximating syllables as avg_word_length / 3.0
    words_per_sent = word_count / sentence_count
    approx_syllables_per_word = max(1.0, avg_word_length / 3.0)
    reading_ease = round(max(0.0, min(100.0, 206.835 - 1.015 * words_per_sent - 84.6 * approx_syllables_per_word)), 1)
    
    if reading_ease >= 70.0:
        reading_level = "Conversational / Easy"
    elif reading_ease >= 50.0:
        reading_level = "Standard Journalistic"
    elif reading_ease >= 30.0:
        reading_level = "Complex / Academic"
    else:
        reading_level = "Dense / Technical"
    
    return {
        "char_count": char_count,
        "word_count": word_count,
        "sentence_count": sentence_count,
        "avg_word_length": avg_word_length,
        "reading_time_mins": reading_time_mins,
        "reading_ease": reading_ease,
        "reading_level": reading_level
    }