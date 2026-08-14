"""
Root Launcher Script for House Price Prediction REST API
"""

import uvicorn
from api.config import settings
from app.logger import logger

if __name__ == "__main__":
    logger.info(
        f"Starting {settings.PROJECT_NAME} on \
        http://{settings.HOST}:{settings.PORT}..."
    )
    uvicorn.run(
        "api.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=True,
    )
    