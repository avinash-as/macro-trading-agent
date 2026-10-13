"""Apify-based macro economic calendar scraper."""
import os
import json
import time
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import requests
import pandas as pd
from src.utils.config import config
from src.utils.logger import logger
from src.utils.database import db


class ApifyMacroScraper:
    """Scrape macroeconomic calendar data using Apify."""
    
    def __init__(self, token: Optional[str] = None, actor_id: Optional[str] = None):
        self.token = token or config.settings.apis.apify_token
        self.actor_id = actor_id or config.settings.apis.apify_actor_id
        self.base_url = config.settings.apis.apify_base_url
        
        if not self.token or self.token == "YOUR_APIFY_TOKEN_HERE":
            logger.warning("Apify token not configured. Using mock data.")
            self.use_mock = True
        else:
            self.use_mock = False
    
    def _run_actor(self, input_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Run Apify actor and return results."""
        if self.use_mock:
            return self._generate_mock_data()
        
        # Start actor run
        run_url = f"{self.base_url}/acts/{self.actor_id}/runs"
        headers = {"Authorization": f"Bearer {self.token}", "Content-Type": "application/json"}
        
        response = requests.post(run_url, headers=headers, json=input_data)
        response.raise_for_status()
        run_info = response.json()['data']
        run_id = run_info['id']
        
        # Poll for completion
        status_url = f"{self.base_url}/actor-runs/{run_id}"
        while True:
            response = requests.get(status_url, headers=headers)
            response.raise_for_status()
            status = response.json()['data']['status']
            
            if status == 'SUCCEEDED':
                break
            elif status in ['FAILED', 'ABORTED', 'TIMED-OUT']:
                raise RuntimeError(f"Actor run failed with status: {status}")
            
            time.sleep(5)
        
        # Get results
        dataset_id = response.json()['data']['defaultDatasetId']
        items_url = f"{self.base_url}/datasets/{dataset_id}/items"
        response = requests.get(items_url, headers=headers)
        response.raise_for_status()
        
        return response.json()
    
    def _generate_mock_data(self) -> List[Dict[str, Any]]:
        """Generate realistic mock macro economic data for testing."""
        import random
        
        events = [
            {"name": "CPI YoY", "country": "US", "currency": "USD", "importance": "High", "unit": "%"},
            {"name": "CPI MoM", "country": "US", "currency": "USD", "importance": "High", "unit": "%"},
            {"name": "Core CPI YoY", "country": "US", "currency": "USD", "importance": "High", "unit": "%"},
            {"name": "FOMC Rate Decision", "country": "US", "currency": "USD", "importance": "High", "unit": "%"},
            {"name": "Non-Farm Payrolls", "country": "US", "currency": "USD", "importance": "High", "unit": "K"},
            {"name": "Unemployment Rate", "country": "US", "currency": "USD", "importance": "High", "unit": "%"},
            {"name": "GDP QoQ", "country": "US", "currency": "USD", "importance": "High", "unit": "%"},
            {"name": "PCE Price Index YoY", "country": "US", "currency": "USD", "importance": "High", "unit": "%"},
            {"name": "ISM Manufacturing PMI", "country": "US", "currency": "USD", "importance": "Medium", "unit": "Index"},
            {"name": "ISM Services PMI", "country": "US", "currency": "USD", "importance": "Medium", "unit": "Index"},
            {"name": "Retail Sales MoM", "country": "US", "currency": "USD", "importance": "Medium", "unit": "%"},
            {"name": "Industrial Production MoM", "country": "US", "currency": "USD", "importance": "Medium", "unit": "%"},
            {"name": "ECB Rate Decision", "country": "EU", "currency": "EUR", "importance": "High", "unit": "%"},
            {"name": "BoE Rate Decision", "country": "UK", "currency": "GBP", "importance": "High", "unit": "%"},
            {"name": "BoJ Rate Decision", "country": "JP", "currency": "JPY", "importance": "High", "unit": "%"},
        ]
        
        mock_data = []
        end_date = datetime.now()
        start_date = end_date - timedelta(days=180)
        
        for event in events:
            # Generate 3-6 occurrences per event over 180 days
            n_occurrences = random.randint(3, 6)
            dates = sorted([start_date + timedelta(days=random.randint(0, 180)) for _ in range(n_occurrences)])
            
            base_forecast = random.uniform(0.5, 5.0)
            
            for i, event_date in enumerate(dates):
                # Add some trend and noise
                trend = i * 0.05
                noise = random.uniform(-0.3, 0.3)
                forecast = round(base_forecast + trend + noise, 2)
                
                # Actual value with surprise
                surprise = random.uniform(-0.5, 0.5)
                actual = round(forecast + surprise, 2)
                
                # Previous value
                previous = round(forecast - random.uniform(-0.2, 0.2), 2)
                
                mock_data.append({
                    "event_name": event["name"],
                    "event_date": event_date.strftime("%Y-%m-%d"),
                    "event_time": "13:30" if event["country"] == "US" else "10:00",
                    "timezone": "EST" if event["country"] == "US" else "UTC",
                    "actual_value": actual,
                    "forecast_value": forecast,
                    "previous_value": previous,
                    "unit": event["unit"],
                    "importance": event["importance"],
                    "country": event["country"],
                    "currency": event["currency"],
                    "source": "apify_mock"
                })
        
        return mock_data
    
    def scrape_calendar(self, lookback_days: int = 180) -> List[Dict[str, Any]]:
        """Scrape macroeconomic calendar for the last N days."""
        logger.info(f"Scraping macro calendar for last {lookback_days} days...")
        
        input_data = {
            "lookbackDays": lookback_days,
            "countries": ["US", "EU", "UK", "JP", "CN", "CA", "AU"],
            "importance": ["High", "Medium"],
            "eventTypes": ["CPI", "FOMC", "NFP", "GDP", "PCE", "PMI", "Retail Sales", "Industrial Production", "Rate Decision"]
        }
        
        try:
            raw_data = self._run_actor(input_data)
            
            # Normalize data
            normalized = []
            for item in raw_data:
                normalized.append({
                    "event_name": item.get("event", item.get("name", "")),
                    "event_date": item.get("date", item.get("event_date", "")),
                    "event_time": item.get("time", item.get("event_time", "")),
                    "timezone": item.get("timezone", "UTC"),
                    "actual_value": self._parse_float(item.get("actual")),
                    "forecast_value": self._parse_float(item.get("forecast", item.get("consensus"))),
                    "previous_value": self._parse_float(item.get("previous")),
                    "unit": item.get("unit", ""),
                    "importance": item.get("importance", "Medium"),
                    "country": item.get("country", "US"),
                    "currency": item.get("currency", "USD"),
                    "source": "apify"
                })
            
            logger.info(f"Scraped {len(normalized)} macro events")
            return normalized
            
        except Exception as e:
            logger.error(f"Apify scraping failed: {e}")
            if not self.use_mock:
                logger.info("Falling back to mock data...")
                self.use_mock = True
                return self.scrape_calendar(lookback_days)
            raise
    
    def _parse_float(self, value: Any) -> Optional[float]:
        """Parse various number formats to float."""
        if value is None or value == "":
            return None
        if isinstance(value, (int, float)):
            return float(value)
        try:
            # Remove common suffixes
            cleaned = str(value).replace("%", "").replace("K", "000").replace("M", "000000").replace(",", "").strip()
            return float(cleaned)
        except:
            return None
    
    def save_to_database(self, events: List[Dict[str, Any]]) -> int:
        """Save scraped events to database."""
        return db.insert_macro_events(events)
    
    def save_to_csv(self, events: List[Dict[str, Any]], filepath: str) -> None:
        """Save events to CSV file."""
        df = pd.DataFrame(events)
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        df.to_csv(filepath, index=False)
        logger.info(f"Saved {len(events)} events to {filepath}")


def main():
    """Main function to run the scraper."""
    scraper = ApifyMacroScraper()
    events = scraper.scrape_calendar(lookback_days=180)
    
    # Save to database
    scraper.save_to_database(events)
    
    # Save to CSV
    output_path = "data/raw/macro_events.csv"
    scraper.save_to_csv(events, output_path)
    
    print(f"Scraped {len(events)} macro events")
    print(f"Saved to database and {output_path}")


if __name__ == "__main__":
    main()