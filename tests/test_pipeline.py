"""Tests for macro trading pipeline."""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
sys.path.append('src')

from utils.config import config
from data_ingestion.apify_scraper import ApifyMacroScraper
from data_ingestion.tavily_scraper import TavilyNewsScraper
from feature_engineering.features import MacroFeatureEngineer
from feature_engineering.macro_summary import MacroSummaryGenerator
from modeling.trainer import MacroModelTrainer, WalkForwardValidator
from utils.database import MacroDatabase


class TestApifyScraper:
    """Test Apify macro scraper."""
    
    def test_mock_data_generation(self):
        """Test mock data generation."""
        scraper = ApifyMacroScraper()
        scraper.use_mock = True
        
        events = scraper.scrape_calendar(lookback_days=30)
        
        assert len(events) > 0
        assert all('event_name' in e for e in events)
        assert all('event_date' in e for e in events)
        assert all('actual_value' in e for e in events)
        assert all('forecast_value' in e for e in events)
    
    def test_event_types(self):
        """Test that key event types are present."""
        scraper = ApifyMacroScraper()
        scraper.use_mock = True
        
        events = scraper.scrape_calendar(lookback_days=180)
        event_names = {e['event_name'] for e in events}
        
        # Check for key economic events
        key_events = {'CPI YoY', 'FOMC Rate Decision', 'Non-Farm Payrolls', 'GDP QoQ'}
        assert any(k in str(event_names) for k in key_events)


class TestTavilyScraper:
    """Test Tavily news scraper."""
    
    def test_mock_news_generation(self):
        """Test mock news generation."""
        scraper = TavilyNewsScraper()
        scraper.use_mock = True
        
        articles = scraper.scrape_economic_news(days_back=7)
        
        assert len(articles) > 0
        assert all('title' in a for a in articles)
        assert all('sentiment_score' in a for a in articles)
        assert all('relevance_score' in a for a in articles)
        assert all(-1 <= a['sentiment_score'] <= 1 for a in articles)
        assert all(0 <= a['relevance_score'] <= 1 for a in articles)


class TestFeatureEngineering:
    """Test feature engineering."""
    
    @pytest.fixture
    def sample_macro_data(self):
        """Create sample macro data."""
        dates = pd.date_range('2024-01-01', periods=20, freq='W')
        events = []
        for i, date in enumerate(dates):
            events.append({
                'event_name': 'CPI YoY' if i % 4 == 0 else 'NFP',
                'event_date': date.strftime('%Y-%m-%d'),
                'actual_value': 3.0 + np.random.randn() * 0.2,
                'forecast_value': 3.0 + np.random.randn() * 0.1,
                'previous_value': 3.0 + np.random.randn() * 0.1,
                'importance': 'High',
                'country': 'US',
                'currency': 'USD',
                'unit': '%'
            })
        return pd.DataFrame(events)
    
    def test_create_event_features(self, sample_macro_data):
        """Test event feature creation."""
        engineer = MacroFeatureEngineer()
        result = engineer.create_event_features(sample_macro_data)
        
        assert 'surprise' in result.columns
        assert 'surprise_direction' in result.columns
        assert 'weighted_surprise' in result.columns
        assert 'importance_score' in result.columns
    
    def test_create_daily_features(self, sample_macro_data):
        """Test daily feature aggregation."""
        engineer = MacroFeatureEngineer()
        event_features = engineer.create_event_features(sample_macro_data)
        
        start_date = '2024-01-01'
        end_date = '2024-05-01'
        daily = engineer.create_daily_features(event_features, start_date, end_date)
        
        assert 'feature_date' in daily.columns
        assert len(daily) > 0
        # Check rolling features exist
        rolling_cols = [c for c in daily.columns if 'rolling_' in c]
        assert len(rolling_cols) > 0
    
    def test_create_time_features(self):
        """Test time feature creation."""
        engineer = MacroFeatureEngineer()
        dates = pd.date_range('2024-01-01', periods=100, freq='D')
        time_features = engineer.create_time_features(dates)
        
        assert 'dow_sin' in time_features.columns
        assert 'dow_cos' in time_features.columns
        assert 'month_sin' in time_features.columns
        assert 'is_business_day' in time_features.columns


class TestMacroSummary:
    """Test macro summary generation."""
    
    def test_generate_outlook_structure(self):
        """Test outlook structure."""
        generator = MacroSummaryGenerator()
        
        # Mock the data loading
        generator.engineer.load_macro_data = lambda s, e: pd.DataFrame()
        generator.engineer.load_news_data = lambda s, e: pd.DataFrame()
        
        outlook = generator.generate_outlook(lookback_days=30)
        
        required_keys = [
            'generated_at', 'period', 'key_themes', 'event_analysis',
            'surprise_analysis', 'sentiment_analysis', 'forward_look',
            'risk_factors', 'trading_implications'
        ]
        
        for key in required_keys:
            assert key in outlook
        
        assert isinstance(outlook['key_themes'], list)
        assert isinstance(outlook['risk_factors'], list)
        assert isinstance(outlook['trading_implications'], dict)


class TestWalkForwardValidator:
    """Test walk-forward validation."""
    
    def test_split_generation(self):
        """Test walk-forward split generation."""
        wfv = WalkForwardValidator(train_window=60, test_window=10, step_size=5)
        splits = wfv.split(100)
        
        assert len(splits) > 0
        for train_idx, test_idx in splits:
            assert len(train_idx) == 60
            assert len(test_idx) == 10
            assert train_idx[-1] < test_idx[0]
    
    def test_insufficient_data(self):
        """Test handling of insufficient data."""
        wfv = WalkForwardValidator(train_window=60, test_window=10, step_size=5)
        splits = wfv.split(50)
        
        assert len(splits) == 0


class TestModelTrainer:
    """Test model trainer."""
    
    @pytest.fixture
    def sample_features(self):
        """Create sample feature matrix with targets."""
        np.random.seed(42)
        n = 200
        dates = pd.date_range('2024-01-01', periods=n, freq='D')
        
        df = pd.DataFrame({'feature_date': dates})
        
        # Add some features
        for i in range(10):
            df[f'feature_{i}'] = np.random.randn(n).cumsum()
        
        # Add target
        df['target_surprise_5d'] = np.random.randn(n) * 0.1
        
        return df
    
    def test_prepare_targets(self, sample_features):
        """Test target preparation."""
        trainer = MacroModelTrainer()
        macro_df = pd.DataFrame()  # Empty macro data
        
        result = trainer.prepare_targets(sample_features, macro_df)
        
        assert f'target_surprise_{trainer.target_horizon}d' in result.columns
        assert f'target_direction_{trainer.target_horizon}d' in result.columns
    
    def test_get_model(self):
        """Test model retrieval."""
        trainer = MacroModelTrainer()
        
        for model_name in ['random_forest', 'xgboost', 'lightgbm', 'linear_regression', 'ridge', 'lasso']:
            model = trainer.get_model(model_name)
            assert model is not None


class TestDatabase:
    """Test database operations."""
    
    def test_database_initialization(self, tmp_path):
        """Test database initialization."""
        db_path = tmp_path / "test.db"
        db = MacroDatabase(str(db_path))
        
        # Check tables exist
        import sqlite3
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = {row[0] for row in cursor.fetchall()}
        conn.close()
        
        expected_tables = {'macro_events', 'news_articles', 'features', 'predictions'}
        assert expected_tables.issubset(tables)
    
    def test_insert_macro_events(self, tmp_path):
        """Test macro event insertion."""
        db_path = tmp_path / "test.db"
        db = MacroDatabase(str(db_path))
        
        events = [
            {
                'event_name': 'CPI YoY',
                'event_date': '2024-01-15',
                'actual_value': 3.1,
                'forecast_value': 3.0,
                'previous_value': 3.0,
                'importance': 'High',
                'country': 'US',
                'currency': 'USD'
            }
        ]
        
        inserted = db.insert_macro_events(events)
        assert inserted == 1
        
        # Query back
        result = db.get_macro_events('2024-01-01', '2024-01-31')
        assert len(result) == 1
        assert result.iloc[0]['event_name'] == 'CPI YoY'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])