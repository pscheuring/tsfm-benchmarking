import logging
import os

from src.constants import (
    DATE_FORMAT,
    LOG_DIR,
    LOG_PATH,
    LOGGING_FORMAT,
    LOGGING_LEVEL,
    PROJECT_LOGGER_NAME,
)


def setup_logging() -> None:
    """
    Initialize logging:
    - Save all logs (DEBUG+) to benchmark.log in a timestamped folder.
    - Show only INFO+ in the console.
    - Only create one log file per run.
    """
    os.makedirs(LOG_DIR, exist_ok=True)

    logger = logging.getLogger(PROJECT_LOGGER_NAME)
    logger.setLevel(logging.DEBUG)

    # Remove old logger
    logger.propagate = False
    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter(LOGGING_FORMAT, datefmt=DATE_FORMAT)

    # File handler: Save all logs (DEBUG+)
    file_handler = logging.FileHandler(LOG_PATH)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    # Stream handler: Show only LOGGING_LEVEL+
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(LOGGING_LEVEL)
    stream_handler.setFormatter(formatter)

    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)


setup_logging()
logger = logging.getLogger(PROJECT_LOGGER_NAME)
