"""
arXiv Scraper
Scrapes academic papers from arXiv.org
"""

import requests
import json
from typing import Dict, List
from datetime import datetime
import sys
from pathlib import Path
import time

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils import setup_logger
from config import RAW_DATA_DIR, SCRAPER_TIMEOUT, SCRAPER_RETRIES

logger = setup_logger(__name__, log_file="arxiv_scraping.log")


class ArxivScraper:
    """Scrape academic papers from arXiv"""
    
    def __init__(self):
        self.base_url = "http://export.arxiv.org/api/query?"
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        logger.info("[OK] arXiv Scraper initialized")
    
    def search_papers(self, query: str, max_results: int = 50) -> List[Dict]:
        """
        Search arXiv for papers
        
        Args:
            query: Search query (e.g., 'machine learning', 'deep learning')
            max_results: Maximum number of papers to retrieve
        
        Returns:
            List of paper dictionaries
        """
        try:
            logger.info(f"[SEARCHING] arXiv for: {query}")
            
            params = {
                'search_query': f'cat:cs.LG AND ({query})',  # Computer Science -> Machine Learning
                'start': 0,
                'max_results': max_results,
                'sortBy': 'relevance',
                'sortOrder': 'descending'
            }
            
            response = requests.get(
                self.base_url,
                params=params,
                headers=self.headers,
                timeout=SCRAPER_TIMEOUT
            )
            response.raise_for_status()
            
            papers = self._parse_arxiv_response(response.text)
            logger.info(f"[OK] Found {len(papers)} papers for: {query}")
            
            return papers
        
        except Exception as e:
            logger.error(f"[ERROR] Failed to search arXiv: {e}")
            return []
    
    def _parse_arxiv_response(self, xml_content: str) -> List[Dict]:
        """Parse arXiv XML response"""
        try:
            import xml.etree.ElementTree as ET
            
            root = ET.fromstring(xml_content)
            papers = []
            
            # arXiv uses Atom feed format
            namespace = {'atom': 'http://www.w3.org/2005/Atom'}
            
            for entry in root.findall('atom:entry', namespace):
                paper = {
                    'title': entry.find('atom:title', namespace).text.strip(),
                    'authors': [author.find('atom:name', namespace).text 
                               for author in entry.findall('atom:author', namespace)],
                    'published': entry.find('atom:published', namespace).text,
                    'summary': entry.find('atom:summary', namespace).text.strip().replace('\n', ' '),
                    'arxiv_id': entry.find('atom:id', namespace).text.split('/abs/')[-1],
                    'url': entry.find('atom:id', namespace).text.replace('http://', 'https://'),
                    'source': 'arxiv',
                    'status': 'success',
                    'content_length': len(entry.find('atom:summary', namespace).text)
                }
                papers.append(paper)
            
            return papers
        
        except Exception as e:
            logger.error(f"[ERROR] Failed to parse arXiv response: {e}")
            return []
    
    def get_multiple_papers(self, queries: List[str], max_results: int = 30) -> List[Dict]:
        """
        Search for multiple topics
        
        Args:
            queries: List of search queries
            max_results: Results per query
        
        Returns:
            List of all papers found
        """
        all_papers = []
        seen_ids = set()
        
        for query in queries:
            logger.info(f"\n[QUERY] Searching: {query}")
            papers = self.search_papers(query, max_results)
            
            for paper in papers:
                arxiv_id = paper.get('arxiv_id', '')
                if arxiv_id not in seen_ids:
                    all_papers.append(paper)
                    seen_ids.add(arxiv_id)
                else:
                    logger.warning(f"[DUPLICATE] Already have: {arxiv_id}")
            
            time.sleep(3)  # Respect arXiv rate limits
        
        logger.info(f"[TOTAL] Collected {len(all_papers)} unique papers")
        return all_papers


def main():
    """Main execution"""
    print("\n" + "="*70)
    print("ARXIV SCRAPER - EDUCATIONAL RAG")
    print("="*70 + "\n")
    
    scraper = ArxivScraper()
    
    # Search queries
    queries = [
        "machine learning",
        "deep learning",
        "neural networks",
        "NLP",
        "transformers"
    ]
    
    # Scrape papers
    papers = scraper.get_multiple_papers(queries, max_results=10)
    
    # Save to JSON
    if papers:
        output_file = RAW_DATA_DIR / "arxiv_papers.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({
                'metadata': {
                    'timestamp': datetime.now().isoformat(),
                    'source': 'arxiv',
                    'total_papers': len(papers)
                },
                'papers': papers
            }, f, indent=2, ensure_ascii=False)
        
        logger.info(f"[OK] Saved {len(papers)} papers to: {output_file}")
        print(f"[OK] Saved {len(papers)} papers to: {output_file}")
    
    # Print summary
    print(f"\n{'='*70}")
    print(f"Total Papers Found: {len(papers)}")
    for paper in papers[:3]:
        print(f"\n  • {paper['title']}")
        print(f"    Authors: {', '.join(paper['authors'][:2])}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()