import os
from dotenv import load_dotenv
from typing import Optional
import logging

load_dotenv()

class Config:
    # Telegram
    TELEGRAM_BOT_TOKEN: str = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    TELEGRAM_CHANNEL_ID: str = os.getenv("TELEGRAM_CHANNEL_ID", "").strip()

    # APIs
    QUOTEX_API_KEY: str = os.getenv("QUOTEX_API_KEY", "").strip()
    TRADINGVIEW_API_KEY: str = os.getenv("TRADINGVIEW_API_KEY", "").strip()
    NEWS_SENTIMENT_API_KEY: str = os.getenv("NEWS_SENTIMENT_API_KEY", "").strip()

    # Selenium
    SELENIUM_DRIVER_PATH: str = os.getenv("SELENIUM_DRIVER_PATH", "").strip()

    # Auto trade flag
    AUTO_TRADE_ENABLED: bool = os.getenv("AUTO_TRADE_ENABLED", "False").strip().lower() in ["true", "1", "yes"]

    # Logging
    LOG_LEVEL: int = getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper(), logging.INFO)

    # AI model path
    MODEL_PATH: str = os.getenv("MODEL_PATH", "models/ai_model_quotex.h5")

    # Indicator parameters
    RSI_PERIOD: int = 14
    BOLLINGER_WINDOW: int = 20
    BOLLINGER_STD_MULTIPLIER: float = 2.0
    ALLIGATOR_JAW_PERIOD: int = 13
    ALLIGATOR_TEETH_PERIOD: int = 8
    ALLIGATOR_LIPS_PERIOD: int = 5
    ICHIMOKU_TENKAN_SEN: int = 9
    ICHIMOKU_KIJUN_SEN: int = 26
    ICHIMOKU_SENKOU_B: int = 52

    # Other settings
    MAX_RETRIES: int = 3

    # Broker endpoints and URLs (placeholders, can be extended)
    QUOTEX_API_URL: str = "https://api.quotex.io/v1"
    TRADINGVIEW_API_URL: str = "https://api.tradingview.com"
    NEWS_SENTIMENT_API_URL: str = "https://api.newssentiment.io/v1"

    @classmethod
    def validate(cls) -> None:
        missing = []
        if not cls.TELEGRAM_BOT_TOKEN:
            missing.append("TELEGRAM_BOT_TOKEN")
        if not cls.TELEGRAM_CHANNEL_ID:
            missing.append("TELEGRAM_CHANNEL_ID")
        if not cls.MODEL_PATH or not os.path.isfile(cls.MODEL_PATH):
            missing.append(f"MODEL_PATH (path: {cls.MODEL_PATH} missing or invalid)")

        # At least one data source API key should be configured (Quotex preferred)
        if not (cls.QUOTEX_API_KEY or cls.TRADINGVIEW_API_KEY or cls.NEWS_SENTIMENT_API_KEY):
            missing.append("At least one API key for QUOTEX_API_KEY, TRADINGVIEW_API_KEY, or NEWS_SENTIMENT_API_KEY")

        if missing:
            raise EnvironmentError(f"Missing or invalid configuration variables: {', '.join(missing)}")

    @classmethod
    def refresh_quotex_token(cls) -> Optional[str]:
        # Placeholder for token refresh logic if Quotex API token expires
        # Here, just return current token as no refresh implemented
        return cls.QUOTEX_API_KEY


# Validate config on import
try:
    Config.validate()
except EnvironmentError as e:
    print(f"Configuration error: {e}")
    raise
