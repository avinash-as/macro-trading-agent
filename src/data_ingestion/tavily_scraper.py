"""Tavily-based economic news scraper."""
import os
import json
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import requests
import pandas as pd
from src.utils.config import config
from src.utils.logger import logger
from src.utils.database import db


class TavilyNewsScraper:
    """Scrape economic news using Tavily API."""
    
    def __init__(self, token: Optional[str] = None):
        self.token = token or config.settings.apis.tavily_token
        self.base_url = config.settings.apis.tavily_base_url
        
        if not self.token or self.token == "YOUR_TAVILY_TOKEN_HERE":
            logger.warning("Tavily token not configured. Using mock data.")
            self.use_mock = True
        else:
            self.use_mock = False
    
    def _search(self, query: str, max_results: int = 10, days_back: int = 7) -> List[Dict[str, Any]]:
        """Search Tavily for news articles."""
        if self.use_mock:
            return self._generate_mock_news(query, max_results)
        
        url = f"{self.base_url}/search"
        headers = {"Authorization": f"Bearer {self.token}", "Content-Type": "application/json"}
        
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        
        payload = {
            "query": query,
            "search_depth": "advanced",
            "max_results": max_results,
            "include_domains": config.settings.data_ingestion.news_sources,
            "include_raw_content": True,
            "days": days_back
        }
        
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        
        return response.json().get("results", [])
    
    def _generate_mock_news(self, query: str, max_results: int) -> List[Dict[str, Any]]:
        """Generate mock economic news for testing."""
        import random
        
        templates = [
            "Fed {action} interest rates amid {context}",
            "US {indicator} {direction} more than expected",
            "Inflation {trend} as {indicator} shows {direction}",
            "Market reacts to {indicator} data",
            "Economists revise forecasts after {indicator} report",
            "Central bank signals {policy} stance",
            "{indicator} surprises to the {direction}",
        ]
        
        indicators = ["CPI", "PCE", "NFP", "GDP", "ISM PMI", "Retail Sales", "Industrial Production"]
        actions = ["holds", "raises", "cuts", "signals potential hike", "signals potential cut"]
        contexts = ["inflation concerns", "growth worries", "labor market strength", "financial stability"]
        directions = ["higher", "lower", "above forecast", "below expectations"]
        trends = ["cools", "accelerates", "moderates", "remains elevated"]
        policies = ["hawkish", "dovish", "neutral", "data-dependent"]
        
        mock_results = []
        for i in range(max_results):
            template = random.choice(templates)
            title = template.format(
                action=random.choice(actions),
                context=random.choice(contexts),
                indicator=random.choice(indicators),
                direction=random.choice(directions),
                trend=random.choice(trends),
                policy=random.choice(policies)
            )
            
            # Random date within last 7 days
            pub_date = datetime.now() - timedelta(days=random.randint(0, 7), hours=random.randint(0, 23))
            
            mock_results.append({
                "title": title,
                "url": f"https://example.com/news/{i}",
                "content": f"Detailed content about {title}. " * 3,
                "published_date": pub_date.isoformat(),
                "source": random.choice(config.settings.data_ingestion.news_sources),
                "score": random.uniform(0.5, 1.0)
            })
        
        return mock_results
    
    def scrape_economic_news(self, days_back: int = 7, max_per_query: int = 10) -> List[Dict[str, Any]]:
        """Scrape recent economic news."""
        logger.info(f"Scraping economic news for last {days_back} days...")
        
        queries = [
            "Federal Reserve monetary policy interest rates",
            "US inflation CPI PCE data",
            "US employment non-farm payrolls unemployment",
            "US GDP growth economic outlook",
            "FOMC meeting minutes federal reserve",
            "central bank policy ECB BoE BoJ",
            "market reaction economic data releases",
            "recession risk economic indicators"
        ]
        
        all_articles = []
        seen_urls = set()
        
        for query in queries:
            try:
                results = self._search(query, max_results=max_per_query, days_back=days_back)
                
                for item in results:
                    url = item.get("url", "")
                    if url in seen_urls:
                        continue
                    seen_urls.add(url)
                    
                    # Calculate relevance score based on keyword matching
                    relevance = self._calculate_relevance(item.get("title", "") + " " + item.get("content", ""))
                    
                    # Simple sentiment analysis
                    sentiment = self._analyze_sentiment(item.get("title", "") + " " + item.get("content", ""))
                    
                    # Extract keywords
                    keywords = self._extract_keywords(item.get("title", "") + " " + item.get("content", ""))
                    
                    all_articles.append({
                        "title": item.get("title", ""),
                        "url": url,
                        "content": item.get("content", item.get("raw_content", ""))[:5000],
                        "summary": item.get("content", item.get("raw_content", ""))[:500],
                        "published_at": item.get("published_date", datetime.now().isoformat()),
                        "source": item.get("source", ""),
                        "relevance_score": relevance,
                        "keywords": json.dumps(keywords),
                        "sentiment_score": sentiment
                    })
                    
            except Exception as e:
                logger.warning(f"Query '{query}' failed: {e}")
                if not self.use_mock:
                    logger.info("Falling back to mock data...")
                    self.use_mock = True
                    return self.scrape_economic_news(days_back, max_per_query)
        
        logger.info(f"Scraped {len(all_articles)} unique news articles")
        return all_articles
    
    def _calculate_relevance(self, text: str) -> float:
        """Calculate relevance score based on economic keywords."""
        keywords = [
            "fed", "federal reserve", "fomc", "interest rate", "inflation", "cpi", "pce",
            "employment", "unemployment", "nfp", "payroll", "gdp", "growth", "recession",
            "monetary policy", "central bank", "ecb", "boe", "boj", "rate hike", "rate cut",
            "yield", "treasury", "bond", "dollar", "currency", "trade", "tariff"
        ]
        
        text_lower = text.lower()
        matches = sum(1 for kw in keywords if kw in text_lower)
        return min(matches / len(keywords) * 2, 1.0)
    
    def _analyze_sentiment(self, text: str) -> float:
        """Simple sentiment analysis (-1 to 1)."""
        positive_words = ["strong", "beat", "above", "rise", "gain", "growth", "optimistic", "bullish", "improve", "better", "higher", "strong", "robust", "solid"]
        negative_words = ["weak", "miss", "below", "fall", "loss", "decline", "pessimistic", "bearish", "worse", "lower", "deteriorate", "concern", "risk", "slow"]
        
        text_lower = text.lower()
        pos_count = sum(1 for w in positive_words if w in text_lower)
        neg_count = sum(1 for w in negative_words if w in text_lower)
        
        total = pos_count + neg_count
        if total == 0:
            return 0.0
        return (pos_count - neg_count) / total
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract economic keywords from text."""
        keywords = [
            "fed", "fomc", "cpi", "pce", "nfp", "gdp", "ism", "pmi", "retail sales",
            "industrial production", "unemployment", "inflation", "interest rate",
            "monetary policy", "rate hike", "rate cut", "yield curve", "treasury",
            "dollar index", "recession", "soft landing", "hard landing"
        ]
        
        text_lower = text.lower()
        found = [kw for kw in keywords if kw in text_lower]
        return found[:10]  # Limit to top 10
    
    def save_to_database(self, articles: List[Dict[str, Any]]) -> int:
        """Save scraped articles to database."""
        return db.insert_news_articles(articles)
    
    def save_to_csv(self, articles: List[Dict[str, Any]], filepath: str) -> None:
        """Save articles to CSV file."""
        df = pd.DataFrame(articles)
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        df.to_csv(filepath, index=False)
        logger.info(f"Saved {len(articles)} articles to {filepath}")


def main():
    """Main function to run the news scraper."""
    scraper = TavilyNewsScraper()
    articles = scraper.scrape_economic_news(days_back=7, max_per_query=10)
    
    # Save to database
    scraper.save_to_database(articles)
    
    # Save to CSV
    output_path = "data/raw/economic_news.csv"
    scraper.save_to_csv(articles, output_path)
    
    print(f"Scraped {len(articles)} economic news articles")
    print(f"Saved to database and {output_path}")


if __name__ == "__main__":
    main()