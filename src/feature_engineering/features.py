"""Feature engineering for macro trading signals."""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from src.utils.config import config
from src.utils.logger import logger
from src.utils.database import db


class MacroFeatureEngineer:
    """Create quantitative features from macro data."""
    
    def __init__(self):
        self.time_windows = config.settings.feature_engineering.time_windows
        self.surprise_threshold = config.settings.feature_engineering.surprise_threshold
        self.rolling_windows = config.settings.feature_engineering.rolling_windows
        self.lag_periods = config.settings.feature_engineering.lag_periods
    
    def load_macro_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Load macro events from database."""
        return db.get_macro_events(start_date, end_date)
    
    def load_news_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Load news articles from database."""
        return db.get_news_articles(start_date, end_date)
    
    def calculate_surprise(self, actual: float, forecast: float, previous: float) -> float:
        """Calculate standardized surprise."""
        if forecast is None or forecast == 0:
            return 0.0
        return (actual - forecast) / abs(forecast)
    
    def calculate_momentum(self, series: pd.Series, window: int) -> pd.Series:
        """Calculate momentum over window."""
        return series.diff(window)
    
    def calculate_rolling_stats(self, series: pd.Series, windows: List[int]) -> pd.DataFrame:
        """Calculate rolling statistics for multiple windows."""
        result = pd.DataFrame(index=series.index)
        for w in windows:
            result[f'rolling_mean_{w}'] = series.rolling(w, min_periods=1).mean()
            result[f'rolling_std_{w}'] = series.rolling(w, min_periods=1).std()
            result[f'rolling_min_{w}'] = series.rolling(w, min_periods=1).min()
            result[f'rolling_max_{w}'] = series.rolling(w, min_periods=1).max()
            result[f'rolling_zscore_{w}'] = (series - result[f'rolling_mean_{w}']) / result[f'rolling_std_{w}'].replace(0, np.nan)
        return result
    
    def create_event_features(self, macro_df: pd.DataFrame) -> pd.DataFrame:
        """Create features from macro events."""
        if macro_df.empty:
            return pd.DataFrame()
        
        df = macro_df.copy()
        df['event_date'] = pd.to_datetime(df['event_date'])
        df = df.sort_values('event_date')
        
        # Calculate surprise
        df['surprise'] = df.apply(
            lambda row: self.calculate_surprise(
                row['actual_value'], row['forecast_value'], row['previous_value']
            ), axis=1
        )
        
        # Surprise direction
        df['surprise_direction'] = np.sign(df['surprise'])
        
        # Surprise magnitude categories
        df['surprise_magnitude'] = pd.cut(
            df['surprise'].abs(),
            bins=[0, 0.05, 0.1, 0.2, np.inf],
            labels=['tiny', 'small', 'medium', 'large']
        )
        
        # Event importance encoding
        importance_map = {'High': 3, 'Medium': 2, 'Low': 1}
        df['importance_score'] = df['importance'].map(importance_map).fillna(2)
        
        # Weighted surprise
        df['weighted_surprise'] = df['surprise'] * df['importance_score']
        
        # Time since last event (per event type)
        df['days_since_last_same_event'] = df.groupby('event_name')['event_date'].diff().dt.days
        
        # Cumulative surprises per event type
        df['cum_surprise_event'] = df.groupby('event_name')['surprise'].cumsum()
        
        # Rolling surprise statistics per event type
        for w in self.rolling_windows:
            df[f'surprise_rolling_mean_{w}'] = df.groupby('event_name')['surprise'].transform(
                lambda x: x.rolling(w, min_periods=1).mean()
            )
            df[f'surprise_rolling_std_{w}'] = df.groupby('event_name')['surprise'].transform(
                lambda x: x.rolling(w, min_periods=1).std()
            )
        
        return df
    
    def create_daily_features(self, macro_df: pd.DataFrame, 
                              start_date: str, end_date: str) -> pd.DataFrame:
        """Aggregate event features to daily frequency."""
        if macro_df.empty:
            # Create empty daily frame
            dates = pd.date_range(start_date, end_date, freq='D')
            return pd.DataFrame({'feature_date': dates})
        
        # Create daily date range
        dates = pd.date_range(start_date, end_date, freq='D')
        daily = pd.DataFrame({'feature_date': dates})
        
        # Aggregate events by day
        macro_df['event_date'] = pd.to_datetime(macro_df['event_date'])
        daily_events = macro_df.groupby('event_date').agg({
            'surprise': ['sum', 'mean', 'std', 'count'],
            'weighted_surprise': ['sum', 'mean'],
            'importance_score': 'max',
            'surprise_direction': 'sum',
            'event_name': 'count'
        }).reset_index()
        
        # Flatten column names
        daily_events.columns = ['_'.join(col).strip('_') if col[1] else col[0] 
                                for col in daily_events.columns.values]
        daily_events = daily_events.rename(columns={'event_date_': 'feature_date'})
        
        # Merge with daily frame
        daily = daily.merge(daily_events, on='feature_date', how='left')
        
        # Fill missing days with 0 (no events)
        event_cols = [c for c in daily.columns if c != 'feature_date']
        daily[event_cols] = daily[event_cols].fillna(0)
        
        # Create rolling features on daily aggregates
        for col in ['surprise_sum', 'weighted_surprise_sum', 'surprise_direction_sum']:
            if col in daily.columns:
                rolling_stats = self.calculate_rolling_stats(daily[col], self.rolling_windows)
                daily = pd.concat([daily, rolling_stats], axis=1)
        
        # Lag features
        for lag in self.lag_periods:
            for col in ['surprise_sum', 'weighted_surprise_sum', 'surprise_direction_sum']:
                if col in daily.columns:
                    daily[f'{col}_lag_{lag}'] = daily[col].shift(lag)
        
        return daily
    
    def create_news_features(self, news_df: pd.DataFrame, 
                             start_date: str, end_date: str) -> pd.DataFrame:
        """Create features from news sentiment."""
        if news_df.empty:
            dates = pd.date_range(start_date, end_date, freq='D')
            return pd.DataFrame({'feature_date': dates})
        
        df = news_df.copy()
        df['published_at'] = pd.to_datetime(df['published_at']).dt.date
        df['feature_date'] = pd.to_datetime(df['published_at'])
        
        # Daily news aggregations
        daily_news = df.groupby('feature_date').agg({
            'sentiment_score': ['mean', 'std', 'sum', 'count'],
            'relevance_score': ['mean', 'max'],
        }).reset_index()
        
        daily_news.columns = ['_'.join(col).strip('_') if col[1] else col[0] 
                              for col in daily_news.columns.values]
        
        # Create full date range
        dates = pd.date_range(start_date, end_date, freq='D')
        daily = pd.DataFrame({'feature_date': dates})
        daily = daily.merge(daily_news, on='feature_date', how='left')
        
        # Fill missing
        news_cols = [c for c in daily.columns if c != 'feature_date']
        daily[news_cols] = daily[news_cols].fillna(0)
        
        # Rolling sentiment features
        for col in ['sentiment_score_mean', 'sentiment_score_sum']:
            if col in daily.columns:
                rolling_stats = self.calculate_rolling_stats(daily[col], self.rolling_windows)
                daily = pd.concat([daily, rolling_stats], axis=1)
        
        # Lag features
        for lag in self.lag_periods:
            for col in ['sentiment_score_mean', 'sentiment_score_sum']:
                if col in daily.columns:
                    daily[f'{col}_lag_{lag}'] = daily[col].shift(lag)
        
        return daily
    
    def create_time_features(self, dates: pd.DatetimeIndex) -> pd.DataFrame:
        """Create time-based features."""
        df = pd.DataFrame({'feature_date': dates})
        
        # Calendar features
        df['day_of_week'] = df['feature_date'].dt.dayofweek
        df['day_of_month'] = df['feature_date'].dt.day
        df['week_of_year'] = df['feature_date'].dt.isocalendar().week
        df['month'] = df['feature_date'].dt.month
        df['quarter'] = df['feature_date'].dt.quarter
        df['year'] = df['feature_date'].dt.year
        
        # Cyclical encoding
        df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_month'] / 31)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_month'] / 31)
        
        # Business day indicator
        df['is_business_day'] = (df['day_of_week'] < 5).astype(int)
        
        # Month start/end
        df['is_month_start'] = df['feature_date'].dt.is_month_start.astype(int)
        df['is_month_end'] = df['feature_date'].dt.is_month_end.astype(int)
        df['is_quarter_start'] = df['feature_date'].dt.is_quarter_start.astype(int)
        df['is_quarter_end'] = df['feature_date'].dt.is_quarter_end.astype(int)
        
        return df
    
    def build_feature_matrix(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Build complete feature matrix."""
        logger.info(f"Building feature matrix from {start_date} to {end_date}")
        
        # Load data
        macro_df = self.load_macro_data(start_date, end_date)
        news_df = self.load_news_data(start_date, end_date)
        
        # Create event-level features
        macro_events = self.create_event_features(macro_df)
        
        # Create daily features
        daily_macro = self.create_daily_features(macro_events, start_date, end_date)
        daily_news = self.create_news_features(news_df, start_date, end_date)
        daily_time = self.create_time_features(pd.date_range(start_date, end_date, freq='D'))
        
        # Merge all features
        features = daily_macro.merge(daily_news, on='feature_date', how='outer')
        features = features.merge(daily_time, on='feature_date', how='outer')
        
        # Fill any remaining NaN
        features = features.fillna(0)
        
        # Sort by date
        features = features.sort_values('feature_date').reset_index(drop=True)
        
        logger.info(f"Built feature matrix: {features.shape[0]} days, {features.shape[1]} features")
        return features
    
    def save_features(self, features_df: pd.DataFrame) -> None:
        """Save features to database and CSV."""
        # Convert to long format for database
        long_df = features_df.melt(
            id_vars=['feature_date'], 
            var_name='feature_name', 
            value_name='feature_value'
        )
        long_df['feature_date'] = long_df['feature_date'].dt.strftime('%Y-%m-%d')
        long_df['feature_category'] = 'macro'
        
        db.insert_features(long_df)
        
        # Save to CSV
        output_path = "data/processed/features.csv"
        features_df.to_csv(output_path, index=False)
        logger.info(f"Saved features to {output_path}")


def main():
    """Run feature engineering."""
    engineer = MacroFeatureEngineer()
    
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=180)).strftime("%Y-%m-%d")
    
    features = engineer.build_feature_matrix(start_date, end_date)
    engineer.save_features(features)
    
    print(f"Feature matrix shape: {features.shape}")
    print(f"Features: {list(features.columns)}")


if __name__ == "__main__":
    main()