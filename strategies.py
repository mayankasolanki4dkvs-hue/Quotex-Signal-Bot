from typing import Dict, Tuple, List
import pandas as pd
import numpy as np
from indicators import rsi, alligator, ichimoku_cloud, bollinger_bands, candle_psychology, candle_sentiment, trend_confirmation
from logger import logger


class BaseStrategy:
    name: str = "BaseStrategy"

    def run_strategy(self, market_data: pd.DataFrame) -> Dict[str, float]:
        """
        Must be implemented by subclasses.
        Returns dict:
            {
                "signal": "BUY"/"SELL"/"REVERSE"/"HOLD",
                "confidence": float between 0.8 and 0.95
            }
        """
        raise NotImplementedError


class StrategySniperReversal(BaseStrategy):
    name = "Sniper Reversal"

    def run_strategy(self, market_data: pd.DataFrame) -> Dict[str, float]:
        # RSI oversold/overbought + Support/Resistance (using Bollinger Bands mid)
        try:
            rsi_series = rsi(market_data)
            bb = bollinger_bands(market_data)
            candle_signal = candle_psychology(market_data)
            if rsi_series.empty or bb['middle_band'].empty:
                return {"signal": "HOLD", "confidence": 0.8}

            last_idx = market_data.index[-1]
            rsi_val = rsi_series.loc[last_idx]
            price = market_data['close'].loc[last_idx]
            mid_band = bb['middle_band'].loc[last_idx]
            candle_sig = candle_signal.loc[last_idx]

            if rsi_val < 30 and price < mid_band and candle_sig == 1:
                return {"signal": "BUY", "confidence": 0.9}
            elif rsi_val > 70 and price > mid_band and candle_sig == -1:
                return {"signal": "SELL", "confidence": 0.9}
            else:
                return {"signal": "HOLD", "confidence": 0.8}
        except Exception as e:
            logger.error(f"{self.name} error: {e}")
            return {"signal": "HOLD", "confidence": 0.8}


class StrategyTrendScalp(BaseStrategy):
    name = "Trend Scalp"

    def run_strategy(self, market_data: pd.DataFrame) -> Dict[str, float]:
        # Uses volume + momentum (RSI derivative) to scalp trends
        try:
            if market_data.empty or 'volume' not in market_data.columns:
                return {"signal": "HOLD", "confidence": 0.8}
            rsi_series = rsi(market_data)
            volume = market_data['volume'].fillna(0)
            if len(rsi_series) < 3:
                return {"signal": "HOLD", "confidence": 0.8}
            rsi_diff = rsi_series.diff()
            last_idx = market_data.index[-1]
            vol = volume.loc[last_idx]
            rsi_delta = rsi_diff.loc[last_idx]

            # Volume threshold and momentum
            if vol > volume.mean() and rsi_delta > 5:
                return {"signal": "BUY", "confidence": 0.87}
            elif vol > volume.mean() and rsi_delta < -5:
                return {"signal": "SELL", "confidence": 0.87}
            else:
                return {"signal": "HOLD", "confidence": 0.8}
        except Exception as e:
            logger.error(f"{self.name} error: {e}")
            return {"signal": "HOLD", "confidence": 0.8}


class StrategyBreakoutTrap(BaseStrategy):
    name = "Breakout Trap"

    def run_strategy(self, market_data: pd.DataFrame) -> Dict[str, float]:
        # Fakeout detection: price breaks Bollinger upper/lower band then reverses
        try:
            bb = bollinger_bands(market_data)
            close = market_data['close']
            if bb['upper_band'].empty or bb['lower_band'].empty or close.empty:
                return {"signal": "HOLD", "confidence": 0.8}

            last_idx = market_data.index[-1]
            prev_idx = market_data.index[-2] if len(market_data) >= 2 else last_idx

            price = close.loc[last_idx]
            prev_price = close.loc[prev_idx]
            upper = bb['upper_band'].loc[last_idx]
            lower = bb['lower_band'].loc[last_idx]

            # Fakeout buy trap: price > upper band then closes below upper band next candle
            if prev_price > upper and price < upper:
                return {"signal": "SELL", "confidence": 0.9}
            # Fakeout sell trap: price < lower band then closes above lower band next candle
            elif prev_price < lower and price > lower:
                return {"signal": "BUY", "confidence": 0.9}
            else:
                return {"signal": "HOLD", "confidence": 0.8}
        except Exception as e:
            logger.error(f"{self.name} error: {e}")
            return {"signal": "HOLD", "confidence": 0.8}


class StrategyEMACrossfire(BaseStrategy):
    name = "EMA Crossfire"

    def run_strategy(self, market_data: pd.DataFrame) -> Dict[str, float]:
        # EMA 9 and EMA 21 crossover
        try:
            if market_data.empty:
                return {"signal": "HOLD", "confidence": 0.8}

            close = market_data['close']
            ema9 = close.ewm(span=9, adjust=False).mean()
            ema21 = close.ewm(span=21, adjust=False).mean()

            if len(ema9) < 2 or len(ema21) < 2:
                return {"signal": "HOLD", "confidence": 0.8}

            last_idx = market_data.index[-1]
            prev_idx = market_data.index[-2]

            # Cross up
            if ema9.loc[prev_idx] < ema21.loc[prev_idx] and ema9.loc[last_idx] > ema21.loc[last_idx]:
                return {"signal": "BUY", "confidence": 0.9}
            # Cross down
            elif ema9.loc[prev_idx] > ema21.loc[prev_idx] and ema9.loc[last_idx] < ema21.loc[last_idx]:
                return {"signal": "SELL", "confidence": 0.9}
            else:
                return {"signal": "HOLD", "confidence": 0.8}
        except Exception as e:
            logger.error(f"{self.name} error: {e}")
            return {"signal": "HOLD", "confidence": 0.8}


class StrategyLiquiditySweep(BaseStrategy):
    name = "Liquidity Sweep"

    def run_strategy(self, market_data: pd.DataFrame) -> Dict[str, float]:
        # Detect stop-hunt and reversal candle patterns
        try:
            candle_signal = candle_psychology(market_data)
            if candle_signal.empty:
                return {"signal": "HOLD", "confidence": 0.8}
            last_idx = market_data.index[-1]

            # Bull trap: bearish engulfing after liquidity sweep low
            if candle_signal.loc[last_idx] == -1:
                return {"signal": "SELL", "confidence": 0.9}
            # Bear trap: bullish engulfing after liquidity sweep high
            if candle_signal.loc[last_idx] == 1:
                return {"signal": "BUY", "confidence": 0.9}
            return {"signal": "HOLD", "confidence": 0.8}
        except Exception as e:
            logger.error(f"{self.name} error: {e}")
            return {"signal": "HOLD", "confidence": 0.8}


class StrategyTimeActionSnipe(BaseStrategy):
    name = "Time Action Snipe"

    def run_strategy(self, market_data: pd.DataFrame) -> Dict[str, float]:
        # Astro + session timing based signal stub (returns HOLD here)
        # Could be extended with actual session times and astro data
        return {"signal": "HOLD", "confidence": 0.8}


class StrategyAISmartZones(BaseStrategy):
    name = "AI Smart Zones"

    def run_strategy(self, market_data: pd.DataFrame) -> Dict[str, float]:
        # Auto Support/Resistance detection stub (returns HOLD)
        return {"signal": "HOLD", "confidence": 0.85}


class StrategyNeuralPatternRecognition(BaseStrategy):
    name = "Neural Pattern Recognition"

    def run_strategy(self, market_data: pd.DataFrame) -> Dict[str, float]:
        # Pattern probability > 0.8 stub (returns HOLD)
        return {"signal": "HOLD", "confidence": 0.85}


class StrategyVolatilityPulse(BaseStrategy):
    name = "Volatility Pulse"

    def run_strategy(self, market_data: pd.DataFrame) -> Dict[str, float]:
        # Compression to expansion detection using Bollinger Bands width
        try:
            bb = bollinger_bands(market_data)
            if bb['upper_band'].empty or bb['lower_band'].empty:
                return {"signal": "HOLD", "confidence": 0.8}

            width = bb['upper_band'] - bb['lower_band']
            if len(width) < 3:
                return {"signal": "HOLD", "confidence": 0.8}

            last_idx = market_data.index[-1]
            prev_idx = market_data.index[-2]
            prev_prev_idx = market_data.index[-3]

            # Detect squeeze (compressed bands) then expansion (width increasing)
            if width.loc[prev_prev_idx] < width.loc[prev_idx] and width.loc[prev_idx] < width.loc[last_idx]:
                return {"signal": "BUY", "confidence": 0.87}
            else:
                return {"signal": "HOLD", "confidence": 0.8}
        except Exception as e:
            logger.error(f"{self.name} error: {e}")
            return {"signal": "HOLD", "confidence": 0.8}


class StrategyDynamicScalper(BaseStrategy):
    name = "Dynamic Scalper"

    def run_strategy(self, market_data: pd.DataFrame) -> Dict[str, float]:
        # Adaptive stochastic AI trader stub (returns HOLD)
        return {"signal": "HOLD", "confidence": 0.85}


def run_all_strategies(market_data: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """
    Runs all strategies on the given market data.
    Returns a dict with strategy names as keys and their signals.
    """
    strategies = [
        StrategySniperReversal(),
        StrategyTrendScalp(),
        StrategyBreakoutTrap(),
        StrategyEMACrossfire(),
        StrategyLiquiditySweep(),
        StrategyTimeActionSnipe(),
        StrategyAISmartZones(),
        StrategyNeuralPatternRecognition(),
        StrategyVolatilityPulse(),
        StrategyDynamicScalper(),
    ]

    results = {}
    for strategy in strategies:
        try:
            res = strategy.run_strategy(market_data)
            results[strategy.name] = res
        except Exception as e:
            logger.error(f"Error running strategy {strategy.name}: {e}")
            results[strategy.name] = {"signal": "HOLD", "confidence": 0.8}
    return results


def aggregate_signals(strategy_results: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    """
    Aggregates signals from all strategies using weighted voting based on confidence.
    Returns a dict: {"final_signal": "BUY"/"SELL"/"REVERSE"/"HOLD", "confidence": float}
    """
    signal_weights = {"BUY": 0, "SELL": 0, "REVERSE": 0, "HOLD": 0}

    for result in strategy_results.values():
        sig = result.get("signal", "HOLD").upper()
        conf = result.get("confidence", 0.8)
        if sig not in signal_weights:
            sig = "HOLD"
        signal_weights[sig] += conf

    # Determine final signal by max weighted confidence
    final_signal = max(signal_weights, key=signal_weights.get)
    total_confidence = signal_weights[final_signal]
    # Normalize confidence to 0-1 by dividing by number of strategies
    normalized_confidence = min(1.0, total_confidence / len(strategy_results))

    return {"final_signal": final_signal, "confidence": normalized_confidence}


__all__ = [
    "BaseStrategy",
    "run_all_strategies",
    "aggregate_signals",
]
