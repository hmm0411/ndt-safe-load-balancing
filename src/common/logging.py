import logging
import os

def configure_logging(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    level = getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper(), logging.INFO)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(
            "%(asctime)s %(levelname)s %(name)s %(message)s"
        ))
        logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False
    return logger
