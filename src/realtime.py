"""
Real-time News Ingestion Module.
Fetches live headlines via NewsAPI with keyword/category filters,
and provides a resilient offline cache/fallback.
"""
import os
from typing import List, Dict, Any
import requests

DEFAULT_FALLBACK_HEADLINES = [
    {
        "title": "Federal Reserve maintains benchmark interest rates steady following quarterly economic review.",
        "source": "Financial Chronicle",
        "category": "business"
    },
    {
        "title": "NASA James Webb Space Telescope discovers water vapor atmospheric signatures in exoplanet system.",
        "source": "Science Today",
        "category": "science"
    },
    {
        "title": "Global summit addresses international climate targets and renewable energy infrastructure.",
        "source": "World Dispatch",
        "category": "general"
    },
    {
        "title": "SHOCKING ALIEN CONSPIRACY CONFIRMED: Secret underground bunker found beneath capitol building!",
        "source": "Viral Truth News",
        "category": "sensational"
    },
    {
        "title": "Tech giants unveil next-generation quantum computing chip architecture with high efficiency.",
        "source": "Tech Radar",
        "category": "technology"
    },
    {
        "title": "BREAKING: Miraculous herbal fruit drink cures all known chronic diseases instantly, doctors furious!",
        "source": "Miracle Health Daily",
        "category": "sensational"
    }
]


def fetch_news(
    api_key: str = None, 
    category: str = "general", 
    query: str = None, 
    country: str = "us", 
    page_size: int = 6
) -> List[Dict[str, Any]]:
    """
    Fetches real-time news articles from NewsAPI or returns fallback articles.
    
    Returns:
        List of dicts with keys: 'title', 'source', 'url', 'published_at', 'description'
    """
    # Check passed key or environment variable
    resolved_key = api_key or os.environ.get("NEWS_API_KEY", "d3f79868d24a46cebabd4a40beddbc18")
    
    articles_data = []
    
    if resolved_key:
        try:
            params = {
                "apiKey": resolved_key,
                "pageSize": page_size
            }
            if query and query.strip():
                url = "https://newsapi.org/v2/everything"
                params["q"] = query.strip()
                params["sortBy"] = "publishedAt"
                params["language"] = "en"
            else:
                url = "https://newsapi.org/v2/top-headlines"
                params["country"] = country
                if category and category != "all":
                    params["category"] = category

            response = requests.get(url, params=params, timeout=6)
            if response.status_code == 200:
                payload = response.json()
                for item in payload.get("articles", []):
                    title = item.get("title")
                    if title and title != "[Removed]":
                        articles_data.append({
                            "title": title,
                            "source": item.get("source", {}).get("name", "News Feed"),
                            "url": item.get("url", ""),
                            "published_at": item.get("publishedAt", "")[:10] if item.get("publishedAt") else "",
                            "description": item.get("description", "") or ""
                        })
        except Exception as e:
            # Fallback gracefully
            print(f"NewsAPI error: {e}")
            
    # If no articles were fetched (API error, rate limit, or no internet), use fallback
    if not articles_data:
        for item in DEFAULT_FALLBACK_HEADLINES:
            articles_data.append({
                "title": item["title"],
                "source": item["source"],
                "url": "https://example.com/news",
                "published_at": "Today",
                "description": item["title"]
            })
            
    return articles_data[:page_size]