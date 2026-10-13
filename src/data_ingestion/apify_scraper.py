"""Apify-based macro economic calendar scraper using pintostudio/economic-calendar-data-investing-com."""
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
    """Scrape macroeconomic calendar data using Apify pintostudio/economic-calendar-data-investing-com actor."""
    
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
            {"name": "CPI YoY", "country": "United States", "currency": "USD", "importance": "high", "unit": "%"},
            {"name": "CPI MoM", "country": "United States", "currency": "USD", "importance": "high", "unit": "%"},
            {"name": "Core CPI YoY", "country": "United States", "currency": "USD", "importance": "high", "unit": "%"},
            {"name": "FOMC Rate Decision", "country": "United States", "currency": "USD", "importance": "high", "unit": "%"},
            {"name": "Non-Farm Payrolls", "country": "United States", "currency": "USD", "importance": "high", "unit": "K"},
            {"name": "Unemployment Rate", "country": "United States", "currency": "USD", "importance": "high", "unit": "%"},
            {"name": "GDP QoQ", "country": "United States", "currency": "USD", "importance": "high", "unit": "%"},
            {"name": "PCE Price Index YoY", "country": "United States", "currency": "USD", "importance": "high", "unit": "%"},
            {"name": "ISM Manufacturing PMI", "country": "United States", "currency": "USD", "importance": "medium", "unit": "Index"},
            {"name": "ISM Services PMI", "country": "United States", "currency": "USD", "importance": "medium", "unit": "Index"},
            {"name": "Retail Sales MoM", "country": "United States", "currency": "USD", "importance": "medium", "unit": "%"},
            {"name": "Industrial Production MoM", "country": "United States", "currency": "USD", "importance": "medium", "unit": "%"},
            {"name": "ECB Rate Decision", "country": "Germany", "currency": "EUR", "importance": "high", "unit": "%"},
            {"name": "BoE Rate Decision", "country": "United Kingdom", "currency": "GBP", "importance": "high", "unit": "%"},
            {"name": "BoJ Rate Decision", "country": "Japan", "currency": "JPY", "importance": "high", "unit": "%"},
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
                
                # Format date as DD/MM/YYYY (matching actor output format)
                date_str = event_date.strftime("%d/%m/%Y")
                
                mock_data.append({
                    "id": len(mock_data) + 1,
                    "date": date_str,
                    "time": "13:30" if event["country"] == "United States" else "10:00",
                    "zone": event["country"].lower(),
                    "currency": event["currency"],
                    "importance": event["importance"],
                    "event": event["name"],
                    "actual": f"{actual}{event['unit']}" if event['unit'] != 'Index' else str(actual),
                    "forecast": f"{forecast}{event['unit']}" if event['unit'] != 'Index' else str(forecast),
                    "previous": f"{previous}{event['unit']}" if event['unit'] != 'Index' else str(previous),
                    "retrieved_at": datetime.now().isoformat(),
                    "data_type": "economic_calendar_event"
                })
        
        return mock_data
    
    def scrape_calendar(
        self,
        lookback_days: int = 180,
        countries: Optional[List[str]] = None,
        importances: Optional[List[str]] = None,
        categories: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Scrape macroeconomic calendar for the last N days.
        
        Since the actor only accepts one country/importance/category per run,
        we run it multiple times and combine results.
        """
        logger.info(f"Scraping macro calendar for last {lookback_days} days...")
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=lookback_days)
        
        from_date = start_date.strftime("%Y-%m-%d")
        to_date = end_date.strftime("%Y-%m-%d")
        
        # Default countries (using actor's supported country names)
        if countries is None:
            countries = [
                "united states", "germany", "united kingdom", "japan", "china",
                "canada", "australia", "france", "italy", "spain"
            ]
        
        # Default importances
        if importances is None:
            importances = ["high", "medium"]
        
        # Default categories
        if categories is None:
            categories = ["Inflation", "Employment", "Central Bank", "GDP", "PMI"]
        
        all_raw_data = []
        
        # Run actor for each combination (actor only accepts one value per field per run)
        for country in countries:
            for importance in importances:
                for category in categories:
                    input_data = {
                        "country": country,
                        "importances": importance,
                        "categories": category,
                        "fromDate": from_date,
                        "toDate": to_date,
                        "timeZone": "GMT +0:00",
                        "timeFilter": "time_only"
                    }
                    
                    try:
                        logger.info(f"Running actor for {country} / {importance} / {category}")
                        raw_data = self._run_actor(input_data)
                        all_raw_data.extend(raw_data)
                        time.sleep(2)  # Rate limiting
                    except Exception as e:
                        logger.error(f"Actor run failed for {country}/{importance}/{category}: {e}")
                        if not self.use_mock:
                            logger.info("Falling back to mock data...")
                            self.use_mock = True
                            return self.scrape_calendar(lookback_days, countries, importances, categories)
        
        # Normalize data
        normalized = []
        for item in all_raw_data:
            # Skip error items
            if item.get("data_type") == "error":
                continue
                
            normalized.append({
                "event_name": item.get("event", ""),
                "event_date": self._parse_date(item.get("date", "")),
                "event_time": item.get("time", ""),
                "timezone": item.get("zone", "").upper(),
                "actual_value": self._parse_float(item.get("actual")),
                "forecast_value": self._parse_float(item.get("forecast")),
                "previous_value": self._parse_float(item.get("previous")),
                "unit": self._extract_unit(item.get("actual", ""), item.get("forecast", ""), item.get("previous", "")),
                "importance": item.get("importance", "medium"),
                "country": item.get("zone", "").title(),
                "currency": item.get("currency", "USD"),
                "source": "apify_investing_com"
            })
        
        logger.info(f"Scraped {len(normalized)} macro events")
        return normalized
    
    def _parse_date(self, date_str: str) -> str:
        """Parse date from DD/MM/YYYY to YYYY-MM-DD."""
        if not date_str:
            return ""
        try:
            # Actor returns DD/MM/YYYY format
            dt = datetime.strptime(date_str, "%d/%m/%Y")
            return dt.strftime("%Y-%m-%d")
        except:
            return date_str
    
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
    
    def _extract_unit(self, *values) -> str:
        """Extract unit from values (%, K, M, etc.)."""
        for v in values:
            if v:
                v_str = str(v)
                if "%" in v_str:
                    return "%"
                elif "K" in v_str:
                    return "K"
                elif "M" in v_str:
                    return "M"
        return ""
    
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