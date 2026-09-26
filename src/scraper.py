"""
Web Scraper Module for URL Article Extraction.
Extracts title, meta description, and body paragraphs from web articles.
"""
from typing import Dict, Any
import requests
from bs4 import BeautifulSoup


def extract_article_from_url(url: str, timeout: int = 8) -> Dict[str, Any]:
    """
    Downloads webpage HTML and extracts article title and textual body.
    """
    url = url.strip()
    if not url.startswith(("http://", "https://")):
        url = "https://" + url

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/118.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"
    }

    try:
        response = requests.get(url, headers=headers, timeout=timeout)
        if response.status_code != 200:
            return {
                "success": False,
                "error": f"HTTP {response.status_code}: Unable to reach website.",
                "url": url
            }

        soup = BeautifulSoup(response.text, "html.parser")

        # Strip scripts, styles, navigations, footers
        for tag in soup(["script", "style", "nav", "footer", "header", "aside", "form"]):
            tag.decompose()

        # Extract title
        title = ""
        if soup.title and soup.title.string:
            title = soup.title.string.strip()
        elif soup.find("h1"):
            title = soup.find("h1").get_text().strip()

        # Extract body paragraphs
        paragraphs = [p.get_text().strip() for p in soup.find_all("p")]
        # Filter short noise paragraphs
        valid_paragraphs = [p for p in paragraphs if len(p.split()) > 7]
        full_text = " ".join(valid_paragraphs)

        if not full_text and title:
            full_text = title

        if not full_text:
            return {
                "success": False,
                "error": "No readable article paragraphs found on page.",
                "url": url
            }

        return {
            "success": True,
            "url": url,
            "title": title,
            "text": full_text,
            "paragraph_count": len(valid_paragraphs)
        }

    except requests.exceptions.Timeout:
        return {"success": False, "error": "Request timed out while connecting to the article.", "url": url}
    except Exception as e:
        return {"success": False, "error": f"Failed to extract article: {str(e)}", "url": url}
