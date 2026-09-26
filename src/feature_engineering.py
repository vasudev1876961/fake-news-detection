from sklearn.feature_extraction.text import TfidfVectorizer

def get_vectorizer():
    return TfidfVectorizer(
        max_features=10000,
        ngram_range=(1,2),
        stop_words='english'
    )