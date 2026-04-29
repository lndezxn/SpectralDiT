from __future__ import annotations

import logging
from pathlib import Path

from rich.console import Console
from rich.logging import RichHandler


def get_console() -> Console:
    return Console()


def setup_logger(name: str, log_file: str | Path | None = None) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        if log_file is not None:
            resolved = str(Path(log_file).resolve())
            has_file_handler = any(
                isinstance(handler, logging.FileHandler) and getattr(handler, "baseFilename", None) == resolved
                for handler in logger.handlers
            )
            if not has_file_handler:
                file_handler = logging.FileHandler(resolved, encoding="utf-8")
                file_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
                logger.addHandler(file_handler)
        return logger

    logger.setLevel(logging.INFO)
    handler = RichHandler(rich_tracebacks=True, show_path=False)
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
    if log_file is not None:
        file_handler = logging.FileHandler(Path(log_file), encoding="utf-8")
        file_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
        logger.addHandler(file_handler)
    logger.propagate = False
    return logger
