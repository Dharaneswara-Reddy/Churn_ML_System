import logging
from pathlib import Path

LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

LOG_FILE = LOG_DIR / "system.log"

def get_logger(name : str) -> logging.Logger:
    """
    Create and return a configured logger 
    
    Parameters: 
        name : str 
            Name of the module requesting the logger.
        returns 
            logging.Logger
    """
    
    logger = logging.getLogger(name)
    
    if logger.handlers:
        return logger # prevent duplicate handlers
    
    logger.setLevel(logging.INFO)
    
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )
    
    file_handler = logging.FileHandler(LOG_FILE)
    file_handler.setFormatter(formatter)
    
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger
