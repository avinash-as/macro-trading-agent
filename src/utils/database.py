"""Database utility for storing and retrieving macro data."""
import sqlite3
import pandas as pd
from pathlib import Path
from contextlib import contextmanager
from typing import Optional, List, Dict, Any
from src.utils.config import config
from src.utils.logger import logger


class MacroDatabase:
    """SQLite database for macro economic data."""
    
    def __init__(self, db_path: Optional[str] = None):
        if db_path is None:
            db_path = Path(config.settings.paths.processed_data) / "macro_data.db"
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
    
    def _init_db(self):
        """Initialize database tables."""
        with self._get_connection() as conn:
            # Macro events table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS macro_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_name TEXT NOT NULL,
                    event_date TEXT NOT NULL,  -- ISO format UTC
                    event_time TEXT,           -- Time component
                    timezone TEXT DEFAULT 'UTC',
                    actual_value REAL,
                    forecast_value REAL,
                    previous_value REAL,
                    unit TEXT,
                    importance TEXT,           -- High/Medium/Low
                    country TEXT,
                    currency TEXT,
                    source TEXT,
                    scraped_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(event_name, event_date, source)
                )
            """)
            
            # News articles table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS news_articles (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    title TEXT NOT NULL,
                    url TEXT UNIQUE NOT NULL,
                    content TEXT,
                    summary TEXT,
                    published_at TEXT,         -- ISO format
                    source TEXT,
                    relevance_score REAL,
                    keywords TEXT,             -- JSON array
                    sentiment_score REAL,
                    scraped_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Feature store table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS features (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    feature_date TEXT NOT NULL,  -- ISO format date
                    feature_name TEXT NOT NULL,
                    feature_value REAL,
                    feature_category TEXT,       -- macro, news, technical, etc.
                    UNIQUE(feature_date, feature_name)
                )
            """)
            
            # Model predictions table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    prediction_date TEXT NOT NULL,
                    target_date TEXT NOT NULL,
                    model_name TEXT NOT NULL,
                    strategy_name TEXT,
                    predicted_pnl REAL,
                    predicted_win_rate REAL,
                    confidence_lower REAL,
                    confidence_upper REAL,
                    features_used TEXT,          -- JSON array
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create indexes
            conn.execute("CREATE INDEX IF NOT EXISTS idx_macro_events_date ON macro_events(event_date)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_news_published ON news_articles(published_at)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_features_date ON features(feature_date)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_predictions_date ON predictions(prediction_date)")
            
            conn.commit()
        logger.info(f"Database initialized at {self.db_path}")
    
    @contextmanager
    def _get_connection(self):
        """Context manager for database connections."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()
    
    def insert_macro_events(self, events: List[Dict[str, Any]]) -> int:
        """Insert macro economic events."""
        inserted = 0
        with self._get_connection() as conn:
            for event in events:
                try:
                    conn.execute("""
                        INSERT OR IGNORE INTO macro_events 
                        (event_name, event_date, event_time, timezone, actual_value, 
                         forecast_value, previous_value, unit, importance, country, 
                         currency, source)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        event.get('event_name'),
                        event.get('event_date'),
                        event.get('event_time'),
                        event.get('timezone', 'UTC'),
                        event.get('actual_value'),
                        event.get('forecast_value'),
                        event.get('previous_value'),
                        event.get('unit'),
                        event.get('importance'),
                        event.get('country'),
                        event.get('currency'),
                        event.get('source')
                    ))
                    if conn.total_changes > inserted:
                        inserted += 1
                except Exception as e:
                    logger.warning(f"Failed to insert event: {e}")
            conn.commit()
        logger.info(f"Inserted {inserted} new macro events")
        return inserted
    
    def insert_news_articles(self, articles: List[Dict[str, Any]]) -> int:
        """Insert news articles."""
        inserted = 0
        with self._get_connection() as conn:
            for article in articles:
                try:
                    conn.execute("""
                        INSERT OR IGNORE INTO news_articles 
                        (title, url, content, summary, published_at, source, 
                         relevance_score, keywords, sentiment_score)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        article.get('title'),
                        article.get('url'),
                        article.get('content'),
                        article.get('summary'),
                        article.get('published_at'),
                        article.get('source'),
                        article.get('relevance_score'),
                        article.get('keywords'),
                        article.get('sentiment_score')
                    ))
                    if conn.total_changes > inserted:
                        inserted += 1
                except Exception as e:
                    logger.warning(f"Failed to insert article: {e}")
            conn.commit()
        logger.info(f"Inserted {inserted} new news articles")
        return inserted
    
    def insert_features(self, features_df: pd.DataFrame) -> int:
        """Insert feature dataframe."""
        inserted = 0
        with self._get_connection() as conn:
            for _, row in features_df.iterrows():
                try:
                    conn.execute("""
                        INSERT OR REPLACE INTO features 
                        (feature_date, feature_name, feature_value, feature_category)
                        VALUES (?, ?, ?, ?)
                    """, (
                        row['feature_date'],
                        row['feature_name'],
                        row['feature_value'],
                        row.get('feature_category', 'macro')
                    ))
                    inserted += 1
                except Exception as e:
                    logger.warning(f"Failed to insert feature: {e}")
            conn.commit()
        logger.info(f"Inserted/updated {inserted} features")
        return inserted
    
    def get_macro_events(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Get macro events within date range."""
        with self._get_connection() as conn:
            query = """
                SELECT * FROM macro_events 
                WHERE event_date BETWEEN ? AND ?
                ORDER BY event_date
            """
            return pd.read_sql_query(query, conn, params=(start_date, end_date))
    
    def get_news_articles(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Get news articles within date range."""
        with self._get_connection() as conn:
            query = """
                SELECT * FROM news_articles 
                WHERE published_at BETWEEN ? AND ?
                ORDER BY published_at DESC
            """
            return pd.read_sql_query(query, conn, params=(start_date, end_date))
    
    def get_features(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Get features within date range as wide format."""
        with self._get_connection() as conn:
            query = """
                SELECT * FROM features 
                WHERE feature_date BETWEEN ? AND ?
                ORDER BY feature_date, feature_name
            """
            df = pd.read_sql_query(query, conn, params=(start_date, end_date))
            if df.empty:
                return pd.DataFrame()
            # Pivot to wide format
            return df.pivot_table(
                index='feature_date', 
                columns='feature_name', 
                values='feature_value',
                aggfunc='last'
            ).reset_index()
    
    def get_latest_features(self, n_days: int = 30) -> pd.DataFrame:
        """Get latest N days of features."""
        with self._get_connection() as conn:
            query = """
                SELECT * FROM features 
                WHERE feature_date >= date('now', ?)
                ORDER BY feature_date, feature_name
            """
            df = pd.read_sql_query(query, conn, params=(f'-{n_days} days',))
            if df.empty:
                return pd.DataFrame()
            return df.pivot_table(
                index='feature_date', 
                columns='feature_name', 
                values='feature_value',
                aggfunc='last'
            ).reset_index()


# Global database instance
db = MacroDatabase()