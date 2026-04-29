"""
Scrape all 99 GDPR articles from gdpr-info.eu and save to gdpr_articles.json.
Run once before ingestion: python scraper.py
"""

import json
import time
import requests
from bs4 import BeautifulSoup

ARTICLE_URL = "https://gdpr-info.eu/art-{n}-gdpr/"
OUTPUT_FILE = "gdpr_articles.json"
REQUEST_DELAY = 0.8  # seconds between requests — be polite


def scrape_article(n: int) -> dict | None:
    url = ARTICLE_URL.format(n=n)
    try:
        resp = requests.get(url, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
    except requests.RequestException as e:
        print(f"  Error fetching Article {n}: {e}")
        return None

    if resp.status_code == 404:
        return None
    if resp.status_code != 200:
        print(f"  HTTP {resp.status_code} for Article {n}")
        return None

    soup = BeautifulSoup(resp.text, "html.parser")

    # Title: usually <h1 class="entry-title"> or first <h1>
    title_el = soup.find("h1", class_="entry-title") or soup.find("h1")
    title = title_el.get_text(strip=True) if title_el else f"Article {n}"

    # Main content div
    content_div = (
        soup.find("div", class_="entry-content")
        or soup.find("div", class_="content")
        or soup.find("main")
    )
    if not content_div:
        print(f"  No content div found for Article {n}")
        return None

    # Remove noise: navigation, scripts, ads
    for tag in content_div.find_all(["nav", "script", "style", "aside", "footer"]):
        tag.decompose()

    # Collect paragraphs preserving structure
    paragraphs = []
    for el in content_div.find_all(["p", "li", "h2", "h3"]):
        text = el.get_text(" ", strip=True)
        if text and len(text) > 20:
            paragraphs.append(text)

    if not paragraphs:
        return None

    full_text = f"Article {n}: {title}\n\n" + "\n\n".join(paragraphs)

    return {
        "article_number": n,
        "title": title,
        "url": url,
        "text": full_text,
        "paragraphs": paragraphs,
    }


def scrape_all() -> None:
    articles = []
    for n in range(1, 100):
        print(f"Scraping Article {n}/99 ...", end=" ")
        article = scrape_article(n)
        if article:
            articles.append(article)
            print(f"OK — {article['title'][:60]}")
        else:
            print("skipped")
        time.sleep(REQUEST_DELAY)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(articles, f, ensure_ascii=False, indent=2)

    print(f"\nDone. Saved {len(articles)} articles to {OUTPUT_FILE}")


if __name__ == "__main__":
    scrape_all()
