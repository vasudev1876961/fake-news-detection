"""
Text preprocessing and text analysis utility module.
Maintains exact compatibility with pre-trained models while adding robust edge-case handling.
"""
import re
import string


def clean_text(text: str) -> str:
    """
    Cleans raw text for TF-IDF vectorization and ML model inference.
    Handles HTML tags, URLs, brackets, punctuation, numbers, and multiple whitespaces.
    """
    if text is None:
        return ""
    
    text = str(text)
    text = text.lower()
    
    # Strip bracketed citations/text [e.g., [Reuters]]
    text = re.sub(r'\[.*?\]', '', text)
    
    # Strip web URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    
    # Strip HTML tags
    text = re.sub(r'<.*?>+', '', text)
    
    # Strip newlines
    text = re.sub(r'[\r\n]+', ' ', text)
    
    # Strip words containing digits
    text = re.sub(r'\w*\d\w*', '', text)
    
    # Strip punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Normalize whitespaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


def get_text_statistics(text: str) -> dict:
    """
    Computes key readability and structural metrics for text diagnostics.
    """
    if not text:
        return {
            "char_count": 0,
            "word_count": 0,
            "sentence_count": 0,
            "avg_word_length": 0.0,
            "reading_time_mins": 0.0
        }
    
    raw = str(text).strip()
    words = [w for w in re.split(r'\s+', raw) if w]
    sentences = [s for s in re.split(r'[.!?]+', raw) if s.strip()]
    
    word_count = len(words)
    char_count = len(raw)
    sentence_count = max(1, len(sentences))
    
    avg_word_length = round(sum(len(w) for w in words) / max(1, word_count), 2)
    # Average reading speed ~ 200 words per minute
    reading_time_mins = round(word_count / 200.0, 1)
    
    return {
        "char_count": char_count,
        "word_count": word_count,
        "sentence_count": sentence_count,
        "avg_word_length": avg_word_length,
        "reading_time_mins": reading_time_mins
    }