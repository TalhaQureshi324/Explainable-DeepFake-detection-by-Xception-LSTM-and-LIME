"""Structured logging without recording media content."""

import logging
import sys


def configure_logging(level: str | None = None) -> None:
    import os
    level = level or os.environ.get("DEEPFAKE_LOG_LEVEL", "INFO")
    fmt = "%(asctime)s %(levelname)s %(name)s: %(message)s"
    logging.basicConfig(level=level, format=fmt, stream=sys.stdout)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
