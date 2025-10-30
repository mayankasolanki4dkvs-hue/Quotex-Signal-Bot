import asyncio
import signal
import sys
import datetime
import pandas as pd
import requests
from logger import logger, log_system_health
from config import Config
from telegram_client import telegram_client
from ai_model import ai_model
from indicators import (rsi, alligator, ichimoku_cloud, bollinger_bands,
                        candle_psychology, candle_sentiment, trend_confirmation)
from strategies import run_all_strategies, aggregate_signals
from market_psychology import market_psychology_engine
from image_analyzer import image_analyzer

from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.firefox.service import Service as FirefoxService
from selenium.webdriver.chrome.options import Options as ChromeOptions
from selenium.webdriver.firefox.options import Options as FirefoxOptions
from selenium.common.exceptions import WebDriverException

import threading
import time


class MayankAISignalBot:
    def __init__(self):
        self.auto_trade_enabled = Config.AUTO_TRADE_ENABLED
        self.running = True
        self.loss_streak = 0
        self.session = "London"  # Simplified static session example
        self.broker_api = "quotex"  # Default broker data source
        self.driver = None
        self.driver_lock = threading.Lock()
        self.init_selenium_driver()
        self.init_signal_handlers()

    def init_selenium_driver(self):
        if not self.auto_trade_enabled:
            logger.info("Auto trade disabled, skipping Selenium initialization.")
            return

        driver_path = Config.SELENIUM_DRIVER_PATH
        if not driver_path:
            logger.error("Selenium driver path not configured, disabling auto-trade.")
            self.auto_trade_enabled = False
            return

        try:
            # Attempt Chrome driver first
            chrome_options = ChromeOptions()
            chrome_options.add_argument("--headless")
            chrome_options.add_argument("--disable-gpu")
            self.driver = webdriver.Chrome(service=ChromeService(driver_path), options=chrome_options)
            logger.info("Selenium Chrome driver initialized for auto-trade.")
        except WebDriverException as e:
            logger.warning(f"Chrome driver init failed: {e}, trying Firefox.")
            try:
                firefox_options = FirefoxOptions()
                firefox_options.add_argument("--headless")
                self.driver = webdriver.Firefox(service=FirefoxService(driver_path), options=firefox_options)
                logger.info("Selenium Firefox driver initialized for auto-trade.")
            except WebDriverException as e2:
                logger.error(f"Firefox driver init failed: {e2}")
                self.driver = None
                self.auto_trade_enabled = False
                logger.error("Auto-trade disabled due to Selenium driver failure.")

    def init_signal_handlers(self):
        def shutdown_handler(signum, frame):
            logger.info(f"Received signal {signum}, shutting down gracefully...")
            self.running = False

        signal.signal(signal.SIGINT, shutdown_handler)
        signal.signal(signal.SIGTERM, shutdown_handler)

    def fetch_quotex_data(self) -> pd.DataFrame:
        """
        Fetch latest 1-min OHLCV data from Quotex API.
        Placeholder returns dummy data for demo.
        """
        try:
            # Real implementation would use requests with API key and symbol requests
            # Here, generate dummy data for last 60 minutes
            now = pd.Timestamp.utcnow().floor('T')
            times = pd.date_range(end=now, periods=60, freq='T')
            data = {
                "open": pd.Series(1.1000 + 0.0001 * pd.np.random.randn(60), index=times),
                "high": pd.Series(1.1005 + 0.0001 * pd.np.random.randn(60), index=times),
                "low": pd.Series(1.0995 + 0.0001 * pd.np.random.randn(60), index=times),
                "close": pd.Series(1.1002 + 0.0001 * pd.np.random.randn(60), index=times),
                "volume": pd.Series(100 + 10 * pd.np.random.randn(60), index=times).abs()
            }
            df = pd.DataFrame(data)
            df = df.clip(lower=0.0001)  # Prevent negative prices/volume
            return df
        except Exception as e:
            logger.error(f"Error fetching Quotex data: {e}")
            return pd.DataFrame()

    def fetch_tradingview_data(self) -> pd.DataFrame:
        """
        Fetch 1-min OHLCV data from TradingView API.
        Placeholder returns empty dataframe.
        """
        logger.info("Fetching TradingView data (stub).")
        return pd.DataFrame()

    def fetch_news_sentiment(self) -> float:
        """
        Fetch latest news sentiment score.
        """
        score = market_psychology_engine.fetch_news_sentiment("ALL")
        if score is None:
            logger.warning("No news sentiment data available, defaulting to 0.")
            return 0.0
        return score

    def fetch_backup_broker_data(self) -> pd.DataFrame:
        """
        Fetch data from other brokers.
        Stub returns empty DataFrame.
        """
        logger.info("Fetching backup broker data (stub).")
        return pd.DataFrame()

    def analyze_market(self, df: pd.DataFrame) -> dict:
        """
        Run indicators, strategies, market psychology, AI model, and aggregate signals.
        """
        if df.empty or len(df) < 60:
            logger.warning("Insufficient market data for analysis.")
            return {}

        indicators_summary = {}

        try:
            rsi_series = rsi(df)
            indicators_summary['RSI'] = round(rsi_series.iloc[-1], 2) if not rsi_series.empty else None

            alligator_lines = alligator(df)
            indicators_summary['Cloud'] = "Bullish" if df['close'].iloc[-1] > alligator_lines['jaw'].iloc[-1] else "Bearish"

            bb = bollinger_bands(df)
            close_last = df['close'].iloc[-1]
            if bb['upper_band'].iloc[-1] and close_last > bb['upper_band'].iloc[-1]:
                indicators_summary['Bollinger'] = "Breakout"
            elif bb['lower_band'].iloc[-1] and close_last < bb['lower_band'].iloc[-1]:
                indicators_summary['Bollinger'] = "Breakdown"
            else:
                indicators_summary['Bollinger'] = "Normal"

            candle_signal = candle_psychology(df).iloc[-1]
            indicators_summary['CandleSignal'] = candle_signal

        except Exception as e:
            logger.error(f"Error computing indicators: {e}")

        # Run strategies
        strategy_results = run_all_strategies(df)
        agg_signal = aggregate_signals(strategy_results)

        # Market psychology
        market_mood = market_psychology_engine.market_emotion_classification(df)
        confidence_score = market_psychology_engine.market_confidence_scoring(df)

        # AI prediction
        ai_prediction = ai_model.predict_candle_direction(df)
        ai_signal = ai_prediction.get("prediction", "HOLD")
        ai_confidence = ai_prediction.get("confidence", 0.0)

        # Combine AI prediction confidence into aggregate confidence
        final_confidence = (agg_signal["confidence"] + ai_confidence) / 2

        # Decide final signal: if AI agrees with aggregated strategies and confidence >= 0.95
        final_signal = "HOLD"
        if final_confidence >= 0.95:
            if agg_signal["final_signal"] == ai_signal:
                final_signal = agg_signal["final_signal"]
            else:
                # If disagreement, choose higher confidence
                final_signal = agg_signal["final_signal"] if agg_signal["confidence"] > ai_confidence else ai_signal

        timestamp = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

        # Compose indicators string
        indicators_str = f"RSI={indicators_summary.get('RSI', 'N/A')} | Cloud={indicators_summary.get('Cloud', 'N/A')} | Bollinger={indicators_summary.get('Bollinger', 'N/A')}"

        signal_dict = {
            "pair": "EUR/USD",  # Placeholder pair
            "timeframe": "1 MIN",
            "signal": final_signal,
            "confidence": final_confidence,
            "market_mood": market_mood,
            "indicators": indicators_str,
            "session": self.session,
            "status": "ACTIVE",
            "strategy": ", ".join(strategy_results.keys()),
            "timestamp": timestamp,
        }
        return signal_dict

    async def send_signal_to_telegram(self, signal: dict):
        if not signal or signal.get("signal") == "HOLD":
            logger.info("No actionable signal to send.")
            return
        if signal.get("confidence", 0) < 0.95:
            logger.info(f"Signal confidence {signal.get('confidence'):.2f} below threshold, skipping send.")
            return
        sent = await telegram_client.send_signal(signal)
        if not sent:
            logger.error("Failed to send signal to Telegram.")

    def auto_trade(self, signal: dict):
        """
        Executes trade via Selenium if auto_trading enabled.
        Placeholder implementation.
        """
        if not self.auto_trade_enabled or self.driver is None:
            logger.debug("Auto trade mode disabled or driver unavailable.")
            return

        with self.driver_lock:
            try:
                # Simplified placeholder: open broker site, execute trade based on signal
                logger.info(f"Executing auto-trade for signal {signal.get('signal')} with confidence {signal.get('confidence'):.2f}")
                # TODO: Implement real Selenium trading logic here
                time.sleep(2)  # simulate action delay

                # Update loss streak logic (stub)
                # This would be updated by actual trade result feedback
                self.loss_streak = 0  # reset on success for demo

            except Exception as e:
                logger.error(f"Error during auto trade execution: {e}")

    async def run(self):
        logger.info("Mayank AI Signal Bot started.")
        while self.running:
            try:
                # Fetch market data with failover
                market_data = self.fetch_quotex_data()
                if market_data.empty:
                    logger.warning("Quotex data unavailable, trying TradingView.")
                    market_data = self.fetch_tradingview_data()
                if market_data.empty:
                    logger.warning("TradingView data unavailable, trying backup brokers.")
                    market_data = self.fetch_backup_broker_data()
                if market_data.empty:
                    logger.warning("No market data available, trying image analyzer (stub).")
                    image_signal = image_analyzer.analyze_chart_image("dummy_path.png")
                    if image_signal.get("confidence", 0) >= 0.95:
                        signal_dict = {
                            "pair": "EUR/USD",
                            "timeframe": "1 MIN",
                            "signal": image_signal.get("signal", "HOLD"),
                            "confidence": image_signal.get("confidence", 0),
                            "market_mood": "Unknown",
                            "indicators": "Image Analysis",
                            "session": self.session,
                            "status": "ACTIVE",
                            "strategy": "ImageAnalyzer",
                            "timestamp": datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
                        }
                        await self.send_signal_to_telegram(signal_dict)
                        self.auto_trade(signal_dict)
                    else:
                        logger.info("No high confidence signal from image analyzer.")
                    await asyncio.sleep(60)
                    continue

                signal_dict = self.analyze_market(market_data)
                if signal_dict:
                    await self.send_signal_to_telegram(signal_dict)
                    self.auto_trade(signal_dict)

                await asyncio.sleep(60)  # Run every minute

            except Exception as e:
                logger.error(f"Exception in main loop: {e}")
                await asyncio.sleep(10)  # short wait before retry

        # Graceful shutdown
        if self.driver:
            with self.driver_lock:
                self.driver.quit()
        logger.info("Mayank AI Signal Bot stopped.")


def main():
    bot = MayankAISignalBot()
    loop = asyncio.get_event_loop()

    try:
        loop.run_until_complete(bot.run())
    except (KeyboardInterrupt, SystemExit):
        logger.info("Received exit signal, terminating...")
    finally:
        loop.close()


if __name__ == "__main__":
    main()
