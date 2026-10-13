"""Logging utility for the macro trading pipeline."""
import logging
import os
from pathlib import Path
from src.utils.config import config


def setup_logging(name: str = "macro_pipeline") -> logging.Logger:
    """Set up logging with file and console handlers."""
    log_config = config.settings.logging
    
    # Create log directory
    log_path = Path(log_config.file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, log_config.level.upper()))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Formatter
    formatter = logging.Formatter(log_config.format)
    
    # File handler
    file_handler = logging.FileHandler(log_config.file)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    return logger


# Global logger
logger = setup_logging()