"""Configuration loader for the macro trading pipeline."""
import yaml
import os
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class APIConfig:
    apify_token: str
    apify_actor_id: str
    apify_base_url: str
    tavily_token: str
    tavily_base_url: str
    exa_token: str
    exa_base_url: str
    
    @classmethod
    def from_dict(cls, d: dict):
        """Create from nested dict structure."""
        return cls(
            apify_token=d.get('apify', {}).get('token', ''),
            apify_actor_id=d.get('apify', {}).get('actor_id', ''),
            apify_base_url=d.get('apify', {}).get('base_url', ''),
            tavily_token=d.get('tavily', {}).get('token', ''),
            tavily_base_url=d.get('tavily', {}).get('base_url', ''),
            exa_token=d.get('exa', {}).get('token', ''),
            exa_base_url=d.get('exa', {}).get('base_url', '')
        )


@dataclass
class DataIngestionConfig:
    lookback_days: int
    macro_calendar_sources: list
    news_sources: list
    key_economic_events: list


@dataclass
class FeatureEngineeringConfig:
    time_windows: list
    surprise_threshold: float
    rolling_windows: list
    lag_periods: list


@dataclass
class ModelingConfig:
    target_horizon_days: int
    train_window: int
    test_window: int
    step_size: int
    models: list
    hyperparameter_tuning: dict
    
    @classmethod
    def from_dict(cls, d: dict):
        """Create from nested dict structure."""
        wf = d.get('walk_forward', {})
        return cls(
            target_horizon_days=d.get('target_horizon_days', 5),
            train_window=wf.get('train_window', 60),
            test_window=wf.get('test_window', 10),
            step_size=wf.get('step_size', 5),
            models=d.get('models', []),
            hyperparameter_tuning=d.get('hyperparameter_tuning', {})
        )


@dataclass
class VisualizationConfig:
    chart_style: str
    color_palette: str
    figure_size: list
    dpi: int
    output_formats: list


@dataclass
class PathsConfig:
    raw_data: str
    processed_data: str
    models: str
    charts: str
    reports: str
    logs: str


@dataclass
class LoggingConfig:
    level: str
    format: str
    file: str


@dataclass
class Settings:
    apis: APIConfig
    data_ingestion: DataIngestionConfig
    feature_engineering: FeatureEngineeringConfig
    modeling: ModelingConfig
    visualization: VisualizationConfig
    paths: PathsConfig
    logging: LoggingConfig


class ConfigLoader:
    """Load and manage configuration from YAML file."""
    
    _instance: Optional['ConfigLoader'] = None
    _settings: Optional[Settings] = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if self._settings is None:
            self.load_config()
    
    def load_config(self, config_path: str = "config/settings.yaml") -> Settings:
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        self._settings = Settings(
            apis=APIConfig.from_dict(config_dict['apis']),
            data_ingestion=DataIngestionConfig(**config_dict['data_ingestion']),
            feature_engineering=FeatureEngineeringConfig(**config_dict['feature_engineering']),
            modeling=ModelingConfig.from_dict(config_dict['modeling']),
            visualization=VisualizationConfig(**config_dict['visualization']),
            paths=PathsConfig(**config_dict['paths']),
            logging=LoggingConfig(**config_dict['logging'])
        )
        return self._settings
    
    @property
    def settings(self) -> Settings:
        if self._settings is None:
            self.load_config()
        return self._settings
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get a nested configuration value using dot notation."""
        keys = key.split('.')
        value = self.settings
        for k in keys:
            if hasattr(value, k):
                value = getattr(value, k)
            else:
                return default
        return value


# Global config instance
config = ConfigLoader()