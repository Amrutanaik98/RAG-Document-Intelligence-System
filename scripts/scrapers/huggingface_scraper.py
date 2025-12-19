"""
Hugging Face Documentation Scraper
Scrapes documentation, model cards, and tutorials from Hugging Face
"""

import requests
from bs4 import BeautifulSoup
from typing import Dict, List
from datetime import datetime
import sys
from pathlib import Path
import json
import time

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils import setup_logger
from config import RAW_DATA_DIR, SCRAPER_TIMEOUT, SCRAPER_USER_AGENT

logger = setup_logger(__name__, log_file="huggingface_scraping.log")


class HuggingFaceScraper:
    """Scrape documentation and guides from Hugging Face"""
    
    def __init__(self):
        self.headers = {
            'User-Agent': SCRAPER_USER_AGENT
        }
        logger.info("[OK] Hugging Face Scraper initialized")
    
    def scrape_documentation(self, url: str) -> Dict:
        """
        Scrape Hugging Face documentation page
        
        Args:
            url: Documentation URL
        
        Returns:
            Dictionary with content
        """
        try:
            logger.info(f"[SCRAPING] HF Docs: {url}")
            
            response = requests.get(
                url,
                headers=self.headers,
                timeout=SCRAPER_TIMEOUT
            )
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove scripts and styles
            for script in soup(['script', 'style']):
                script.decompose()
            
            # Get title
            title_elem = soup.find('h1')
            title = title_elem.get_text(strip=True) if title_elem else "No Title"
            
            # Get main content
            main_elem = soup.find('main') or soup.find('article')
            if main_elem:
                content = main_elem.get_text(separator='\n', strip=True)
            else:
                content = soup.get_text(separator='\n', strip=True)
            
            result = {
                'title': title,
                'url': url,
                'content': content,
                'source': 'huggingface',
                'scraped_at': datetime.now().isoformat(),
                'status': 'success',
                'content_length': len(content)
            }
            
            logger.info(f"[OK] Scraped: {title} ({len(content)} chars)")
            return result
        
        except Exception as e:
            logger.error(f"[ERROR] Failed to scrape {url}: {e}")
            return {
                'url': url,
                'status': 'failed',
                'error': str(e),
                'scraped_at': datetime.now().isoformat()
            }
    
    def get_documentation_pages(self, library: str = "transformers") -> List[str]:
        """
        Get documentation page URLs
        
        Args:
            library: Library to scrape ('transformers', 'datasets', 'diffusers', etc)
        
        Returns:
            List of documentation URLs
        """
        base_url = f"https://huggingface.co/docs/{library}"
        
        pages = [
            f"{base_url}/installation",
            f"{base_url}/quicktour",
            f"{base_url}/pipeline_tutorial",
            f"{base_url}/preprocessing",
            f"{base_url}/training",
            f"{base_url}/model_sharing",
        ]
        
        return pages
    
    def scrape_library_docs(self, library: str) -> List[Dict]:
        """
        Scrape all documentation for a library
        
        Args:
            library: Library name
        
        Returns:
            List of scraped pages
        """
        logger.info(f"\n[LIBRARY] Scraping: {library}")
        
        pages = self.get_documentation_pages(library)
        results = []
        
        for page_url in pages:
            result = self.scrape_documentation(page_url)
            results.append(result)
            time.sleep(1)  # Rate limiting
        
        logger.info(f"[OK] Scraped {len(results)} pages for: {library}")
        return results
    
    def get_multiple_docs(self, libraries: List[str] = None) -> List[Dict]:
        """
        Scrape multiple libraries
        
        Args:
            libraries: List of libraries to scrape
        
        Returns:
            List of all scraped pages
        """
        if libraries is None:
            libraries = ["transformers", "datasets", "diffusers"]
        
        all_docs = []
        
        for library in libraries:
            docs = self.scrape_library_docs(library)
            all_docs.extend(docs)
            time.sleep(2)
        
        logger.info(f"[TOTAL] Scraped {len(all_docs)} documentation pages")
        return all_docs


def main():
    """Main execution"""
    print("\n" + "="*70)
    print("HUGGING FACE SCRAPER - EDUCATIONAL RAG")
    print("="*70 + "\n")
    
    scraper = HuggingFaceScraper()
    
    # Libraries to scrape
    libraries = ["transformers"]  # Start with transformers
    
    print(f"Scraping {len(libraries)} Hugging Face libraries...\n")
    
    # Scrape docs
    docs = scraper.get_multiple_docs(libraries)
    
    # Save results
    if docs:
        output_file = RAW_DATA_DIR / "huggingface_docs.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({
                'metadata': {
                    'timestamp': datetime.now().isoformat(),
                    'source': 'huggingface',
                    'total_pages': len(docs)
                },
                'pages': docs
            }, f, indent=2, ensure_ascii=False)
        
        logger.info(f"[OK] Saved {len(docs)} pages to: {output_file}")
        print(f"[OK] Saved {len(docs)} pages to: {output_file}")
    
    # Print summary
    print(f"\n{'='*70}")
    print(f"Total Documentation Pages: {len(docs)}")
    for doc in docs[:3]:
        if doc['status'] == 'success':
            print(f"\n  • {doc['title'][:60]}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()