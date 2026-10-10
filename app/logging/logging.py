"""Logging configuration"""
import logging
import sys
from pathlib import Path

# Create logs directory if it doesn't exist
# LOGS_DIR.mkdir(parents=True, exist_ok=True)

def setup_logging(name: str, level: int = logging.INFO) -> logging.Logger:
    """Setup logger with file and console handlers"""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_format = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(console_format)
    
    # # File handler
    # file_handler = logging.FileHandler(LOGS_DIR / f"{name}.log")
    # file_handler.setLevel(level)
    # file_handler.setFormatter(console_format)
    
    logger.addHandler(console_handler)
    # logger.addHandler(file_handler)
    
    return logger
