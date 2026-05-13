"""
Car Brochure Search System — Application Entry Point

Run:
    poetry run uvicorn app:app --reload --host 0.0.0.0 --port 8000
"""

import logging

from src.api.routes import app  # noqa: F401

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# Suppress noisy loggers
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("elasticsearch").setLevel(logging.WARNING)

if __name__ == "__main__":
    import uvicorn
    from config import get_settings

    settings = get_settings()
    uvicorn.run(
        "app:app",
        host=settings.host,
        port=settings.port,
        reload=True,
    )
