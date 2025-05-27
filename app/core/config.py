from pydantic_settings import BaseSettings
from pathlib import Path
import os

class Settings(BaseSettings):
    APP_NAME: str = "Vidyasa"
    APP_VERSION: str = "0.1.0"
    APP_DESCRIPTION: str = "Vidyasa Backend API Service"
    
    # CORS setting
    CORS_ORIGINS: list[str] = [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        
        "http://localhost:5000",
        "http://127.0.0.1:5000",

        "http://localhost:8000",
        "http://127.0.0.1:8000",
    ]

    class ConfigDict:
        env_file = ".env"

settings = Settings()

