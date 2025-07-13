# services/crawling/crawler.py

import requests
from bs4 import BeautifulSoup
import sqlite3
import time
from urllib.parse import urljoin, urlparse
import uuid

def parse_article_content(soup: BeautifulSoup) -> dict:
    try:
        title = soup.find('h1').get_text(strip=True)
        
        content_div = soup.find('div', class_='article-content')
        if not content_div:
            return None

        paragraphs = content_div.find_all('p')
        content = "\n".join([p.get_text(strip=True) for p in paragraphs])
        if title and content:
            return {'title': title, 'text': content}
    except AttributeError:
        return None
    return None

def crawl_website(start_url: str, max_pages: int = 10) -> list[dict]:
    pages_to_visit = {start_url}
    visited_pages = set()
    crawled_articles = []
    base_netloc = urlparse(start_url).netloc
    while pages_to_visit and len(crawled_articles) < max_pages:
        url = pages_to_visit.pop()
        if url in visited_pages:
            continue
        print(f"Crawling: {url}")
        visited_pages.add(url)
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status() 
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            article_data = parse_article_content(soup)
            if article_data:
                crawled_articles.append(article_data)
                print(f"  -> Found article: {article_data['title'][:50]}...")

            for link in soup.find_all('a', href=True):
                absolute_link = urljoin(url, link['href'])
                if urlparse(absolute_link).netloc == base_netloc and absolute_link not in visited_pages:
                    pages_to_visit.add(absolute_link)

            time.sleep(1)

        except requests.RequestException as e:
            print(f"Could not crawl {url}: {e}")

    return crawled_articles

def merge_crawled_data_to_db(crawled_data: list[dict], db_path: str):

    print(f"\nMerging {len(crawled_data)} crawled documents into database: {db_path}...")
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    for article in crawled_data:
        doc_id = f"crawled-{uuid.uuid4()}"
        
        try:
            cur.execute("INSERT INTO docs (doc_id, zahf) VALUES (?, ?)", (doc_id, article['text']))
        except sqlite3.Error as e:
            print(f"Database error: {e}")

    conn.commit()
    conn.close()
    print("✅ Merging complete.")
