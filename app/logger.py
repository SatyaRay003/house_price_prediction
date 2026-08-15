"""Application logging configuration."""

import logging
import sys


def setup_logger():
    """Configure and return the application logger."""
    app_logger = logging.getLogger("HousePriceML")
    app_logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s:%(lineno)d] - %(message)s"
    )

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)

    if not app_logger.handlers:
        app_logger.addHandler(stream_handler)

    return app_logger


logger = setup_logger()
