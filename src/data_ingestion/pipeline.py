"""Data ingestion orchestrator for macro trading pipeline."""
import os
from datetime import datetime, timedelta
from typing import Dict, Any, List
import pandas as pd
from src.utils.config import config
from src.utils.logger import logger
from src.utils.database import db
from src.data_ingestion.apify_scraper import ApifyMacroScraper
from src.data_ingestion.tavily_scraper import TavilyNewsScraper


class DataIngestionPipeline:
    """Orchestrate data ingestion from multiple sources."""
    
    def __init__(self):
        self.apify_scraper = ApifyMacroScraper()
        self.tavily_scraper = TavilyNewsScraper()
    
    def run_full_ingestion(self, lookback_days: int = 180, news_days: int = 30) -> Dict[str, Any]:
        """Run complete data ingestion pipeline."""
        logger.info("Starting full data ingestion pipeline...")
        
        results = {
            "macro_events": 0,
            "news_articles": 0,
            "start_time": datetime.now().isoformat(),
            "errors": []
        }
        
        # 1. Scrape macroeconomic calendar
        try:
            logger.info("Scraping macroeconomic calendar...")
            macro_events = self.apify_scraper.scrape_calendar(lookback_days=lookback_days)
            results["macro_events"] = len(macro_events)
            
            # Save to database
            self.apify_scraper.save_to_database(macro_events)
            
            # Save raw CSV
            self.apify_scraper.save_to_csv(
                macro_events, 
                os.path.join(config.settings.paths.raw_data, "macro_events.csv")
            )
            
        except Exception as e:
            logger.error(f"Macro calendar scraping failed: {e}")
            results["errors.append"](f"Macro calendar: {e}")
        
        # 2. Scrape economic news
        try:
            logger.info("Scraping economic news...")
            news_articles = self.tavily_scraper.scrape_economic_news(days_back=news_days)
            results["news_articles"] = len(news_articles)
            
            # Save to database
            self.tavily_scraper.save_to_database(news_articles)
            
            # Save raw CSV
            self.tavily_scraper.save_to_csv(
                news_articles,
                os.path.join(config.settings.paths.raw_data, "economic_news.csv")
            )
            
        except Exception as e:
            logger.error(f"News scraping failed: {e}")
            results["errors"].append(f"News: {e}")
        
        results["end_time"] = datetime.now().isoformat()
        results["duration_seconds"] = (
            datetime.fromisoformat(results["end_time"]) - 
            datetime.fromisoformat(results["start_time"])
        ).total_seconds()
        
        logger.info(f"Data ingestion complete: {results}")
        return results
    
    def get_ingestion_status(self) -> Dict[str, Any]:
        """Get status of data in database."""
        # Get date ranges
        macro_df = db.get_macro_events("2020-01-01", datetime.now().strftime("%Y-%m-%d"))
        news_df = db.get_news_articles("2020-01-01", datetime.now().strftime("%Y-%m-%d"))
        
        return {
            "macro_events_count": len(macro_df),
            "macro_date_range": {
                "min": macro_df["event_date"].min() if not macro_df.empty else None,
                "max": macro_df["event_date"].max() if not macro_df.empty else None
            },
            "news_articles_count": len(news_df),
            "news_date_range": {
                "min": news_df["published_at"].min() if not news_df.empty else None,
                "max": news_df["published_at"].max() if not news_df.empty else None
            }
        }


def main():
    """Run the data ingestion pipeline."""
    pipeline = DataIngestionPipeline()
    
    # Run ingestion
    results = pipeline.run_full_ingestion(
        lookback_days=config.settings.data_ingestion.lookback_days,
        news_days=30
    )
    
    # Print status
    status = pipeline.get_ingestion_status()
    print("\n=== Ingestion Results ===")
    print(f"Macro Events: {results['macro_events']}")
    print(f"News Articles: {results['news_articles']}")
    print(f"Duration: {results['duration_seconds']:.1f}s")
    if results["errors"]:
        print(f"Errors: {results['errors']}")
    
    print("\n=== Database Status ===")
    print(f"Total Macro Events: {status['macro_events_count']}")
    print(f"Total News Articles: {status['news_articles_count']}")


if __name__ == "__main__":
    main()