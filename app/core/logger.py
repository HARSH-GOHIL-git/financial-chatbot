import logging
import os
from logging.handlers import RotatingFileHandler

# logs/ folder sits at the project root (app/core/ -> app/ -> root)
_BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LOG_DIR = os.path.join(_BASE_DIR, "logs")
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, "app.log")

_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
_initialized = False


def setup_logging(level: str = "INFO") -> None:
    """Configure root logger with console + rotating file handlers.
    Call exactly once at application startup (inside lifespan)."""
    global _initialized
    if _initialized:
        return
    _initialized = True

    log_level = getattr(logging, level.upper(), logging.INFO)
    formatter = logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)

    root = logging.getLogger()
    root.setLevel(log_level)
    root.handlers.clear()

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(log_level)
    ch.setFormatter(formatter)
    root.addHandler(ch)

    # Rotating file handler: 10 MB per file, keep last 5 backups
    fh = RotatingFileHandler(
        LOG_FILE, maxBytes=10 * 1024 * 1024, backupCount=5, encoding="utf-8"
    )
    fh.setLevel(log_level)
    fh.setFormatter(formatter)
    root.addHandler(fh)

    # Silence noisy third-party libraries
    for lib in ("httpx", "httpcore", "chromadb", "urllib3", "hpack", "uvicorn.access"):
        logging.getLogger(lib).setLevel(logging.WARNING)

    logging.getLogger(__name__).info(f"Logging initialised — writing to: {LOG_FILE}")


def get_logger(name: str) -> logging.Logger:
    """Return a named logger. Always pass __name__ as the argument."""
    return logging.getLogger(name)
