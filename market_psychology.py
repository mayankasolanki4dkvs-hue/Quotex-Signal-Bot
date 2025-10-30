import requests
from typing import Dict, Optional
import logging
import pandas as pd
from logger import logger
from config import Config


class MarketPsychology:
    def __init__(self):
        self.news_api_key = Config.NEWS_SENTIMENT_API_KEY
        self.sentiment_api_url = Config.NEWS_SENTIMENT_API_URL
        self.fear_greed_index_url = "https://api.alternative.me/fng/"  # Public API for Fear & Greed Index

    def fetch_news_sentiment(self, symbol: str) -> Optional[float]:
        """
        Fetch news sentiment score for the symbol (range -1 to 1).
        Returns None if failed.
        """
        if not self.news_api_key:
            logger.warning("News Sentiment API key not configured.")
            return None
        try:
            params = {
                "symbol": symbol,
                "apikey": self.news_api_key
            }
            response = requests.get(self.sentiment_api_url, params=params, timeout=5)
            if response.status_code == 200:
                data = response.json()
                score = data.get("sentiment_score")
                if isinstance(score, (float, int)):
                    return float(score)
                else:
                    logger.warning(f"Invalid sentiment score format from news API: {score}")
            else:
                logger.warning(f"News Sentiment API responded with status {response.status_code}")
        except Exception as e:
            logger.error(f"Error fetching news sentiment: {e}")
        return None

    def fetch_fear_greed_index(self) -> Optional[int]:
        """
        Fetch Fear and Greed index (0-100).
        Public free API used.
        """
        try:
            response = requests.get(self.fear_greed_index_url, timeout=5)
            if response.status_code == 200:
                data = response.json()
                if "data" in data and len(data["data"]) > 0:
                    value_str = data["data"][0].get("value")
                    if value_str and value_str.isdigit():
                        return int(value_str)
            else:
                logger.warning(f"Fear & Greed API responded with status {response.status_code}")
        except Exception as e:
            logger.error(f"Error fetching Fear & Greed index: {e}")
        return None

    def candle_emotion_detection(self, df: pd.DataFrame) -> pd.Series:
        """
        Detects aggressive buying, selling, trap candles, rejections.
        Returns a series with values:
            1 for aggressive buying
            -1 for aggressive selling
            0 for neutral
        """
        if df.empty:
            return pd.Series(dtype=int)

        open_ = df['open']
        close = df['close']
        high = df['high']
        low = df['low']

        body = (close - open_).abs()
        candle_range = high - low
        upper_shadow = high - close.where(close > open_, open_)
        lower_shadow = open_.where(close > open_, close) - low

        emotion = pd.Series(0, index=df.index, dtype=int)

        # Aggressive buying: long green candle with small shadows
        cond_aggressive_buy = (close > open_) & \
                             (body >= 0.7 * candle_range) & \
                             (upper_shadow <= 0.1 * body) & \
                             (lower_shadow <= 0.1 * body)
        emotion[cond_aggressive_buy] = 1

        # Aggressive selling: long red candle with small shadows
        cond_aggressive_sell = (close < open_) & \
                              (body >= 0.7 * candle_range) & \
                              (upper_shadow <= 0.1 * body) & \
                              (lower_shadow <= 0.1 * body)
        emotion[cond_aggressive_sell] = -1

        # Trap candles or rejection candles can be defined as candles with long shadows and small bodies
        cond_rejection = (body <= 0.3 * candle_range) & \
                         ((upper_shadow >= 0.5 * candle_range) | (lower_shadow >= 0.5 * candle_range))
        emotion[cond_rejection] = 0  # Neutral but marked for reference

        return emotion

    def market_emotion_classification(self, df: pd.DataFrame) -> str:
        """
        Classify overall market emotion as 'Fear', 'Greed', or 'Neutral'
        Using fear_greed_index and recent candle emotions
        """
        fear_greed = self.fetch_fear_greed_index()
        if fear_greed is None:
            return "Neutral"

        # Thresholds from 0 to 100
        if fear_greed <= 40:
            return "Fear"
        elif fear_greed >= 60:
            return "Greed"
        else:
            return "Neutral"

    def dynamic_capital_safety_mode(self, loss_streak: int) -> bool:
        """
        Enable low risk mode if loss streak >= 3
        """
        return loss_streak >= 3

    def trap_zone_detection(self, df: pd.DataFrame) -> pd.Series:
        """
        Detect bull and bear traps zones based on candle patterns and volume spikes.
        Returns series: 1 for bull trap, -1 for bear trap, 0 neutral.
        """
        if df.empty or 'volume' not in df.columns:
            return pd.Series(dtype=int)

        candle_emotions = self.candle_emotion_detection(df)
        volume = df['volume'].fillna(0)
        trap_zone = pd.Series(0, index=df.index, dtype=int)

        # Simplified trap detection logic:
        # Bull trap: aggressive buying candle followed by bearish engulfing and volume spike
        for i in range(1, len(df)):
            if candle_emotions.iat[i - 1] == 1 and candle_emotions.iat[i] == -1:
                if volume.iat[i] > volume.mean() * 1.5:
                    trap_zone.iat[i] = 1  # Bull trap

        # Bear trap: aggressive selling candle followed by bullish engulfing and volume spike
        for i in range(1, len(df)):
            if candle_emotions.iat[i - 1] == -1 and candle_emotions.iat[i] == 1:
                if volume.iat[i] > volume.mean() * 1.5:
                    trap_zone.iat[i] = -1  # Bear trap
        return trap_zone

    def market_confidence_scoring(self, df: pd.DataFrame) -> float:
        """
        Aggregate confidence score from news sentiment, fear/greed, trap zones.
        Returns value 0-100.
        """
        news_sentiment = self.fetch_news_sentiment("ALL")  # Symbol can be parameterized
        fear_greed = self.fetch_fear_greed_index()
        trap_zone = self.trap_zone_detection(df)

        score = 50.0  # Neutral base

        if news_sentiment is not None:
            # Scale from -1..1 to 0..100
            score += news_sentiment * 25

        if fear_greed is not None:
            # Normalize 0-100 to -25..25 around 50
            score += (fear_greed - 50) * 0.5

        # Trap zone effect: decrease score if traps detected recently
        recent_traps = trap_zone.tail(5).abs().sum()
        if recent_traps > 0:
            score -= min(20, recent_traps * 5)

        # Clamp score 0-100
        score = max(0, min(100, score))
        return score


market_psychology_engine = MarketPsychology()
