import logging
import logging.config
import os
from datetime import datetime
from pathlib import Path

def setup_logging(log_level: str = "INFO", log_dir: str = "./logs"):
    """Setup logging configuration"""
    
    # Create logs directory if it doesn't exist
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    
    # Generate log filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d")
    log_file = f"{log_dir}/speaker_ai_{timestamp}.log"
    
    logging_config = {
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'standard': {
                'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s',
                'datefmt': '%Y-%m-%d %H:%M:%S'
            },
            'detailed': {
                'format': '%(asctime)s [%(levelname)s] %(name)s:%(lineno)d: %(message)s',
                'datefmt': '%Y-%m-%d %H:%M:%S'
            }
        },
        'handlers': {
            'console': {
                'level': log_level,
                'class': 'logging.StreamHandler',
                'formatter': 'standard',
                'stream': 'ext://sys.stdout'
            },
            'file': {
                'level': log_level,
                'class': 'logging.handlers.RotatingFileHandler',
                'filename': log_file,
                'maxBytes': 10485760,  # 10MB
                'backupCount': 5,
                'formatter': 'detailed'
            }
        },
        'loggers': {
            '': {  # root logger
                'handlers': ['console', 'file'],
                'level': log_level,
                'propagate': False
            },
            'uvicorn': {
                'handlers': ['console', 'file'],
                'level': log_level,
                'propagate': False
            },
            'chromadb': {
                'handlers': ['file'],
                'level': 'WARNING',
                'propagate': False
            }
        }
    }
    
    logging.config.dictConfig(logging_config)
    
    # Create logger instance
    logger = logging.getLogger(__name__)
    logger.info("Logging configuration initialized")
    
    return logger

def get_logger(name: str) -> logging.Logger:
    """Get a logger instance"""
    return logging.getLogger(name)