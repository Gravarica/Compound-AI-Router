# Create logging utility module
import os
import logging
from typing import Optional


def setup_logging(level: int = logging.INFO,
                  log_file: Optional[str] = None,
                  name: Optional[str] = None) -> logging.Logger:

    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    handlers = [logging.StreamHandler()]

    if log_file:
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)

        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        handlers.append(file_handler)

    if name is None:
        logging.basicConfig(
            level=level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=handlers
        )
        logger = logging.getLogger()
    else:
        logger = logging.getLogger(name)
        logger.setLevel(level)

        if logger.hasHandlers():
            logger.handlers.clear()

        for handler in handlers:
            handler.setFormatter(formatter)
            logger.addHandler(handler)

    return logger