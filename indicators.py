import pandas as pd
import numpy as np
from typing import Dict

def rsi(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Calculate Relative Strength Index (RSI).
    """
    if df.empty or len(df) < period + 1:
        return pd.Series(dtype=float)
    delta = df['close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi.name = 'RSI'
    return rsi.fillna(50)  # Neutral RSI for initial periods


def alligator(df: pd.DataFrame,
              jaw_period: int = 13,
              teeth_period: int = 8,
              lips_period: int = 5) -> Dict[str, pd.Series]:
    """
    Calculate Alligator indicator lines: Jaw, Teeth, Lips.
    Uses smoothed moving average (SMMA).
    """
    def smma(series: pd.Series, period: int) -> pd.Series:
        smma_series = series.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
        return smma_series

    high = df['high']
    low = df['low']
    median_price = (high + low) / 2

    jaw = smma(median_price.shift(8), jaw_period)
    teeth = smma(median_price.shift(5), teeth_period)
    lips = smma(median_price.shift(3), lips_period)

    return {
        'jaw': jaw,
        'teeth': teeth,
        'lips': lips,
    }


def ichimoku_cloud(df: pd.DataFrame,
                   tenkan_period: int = 9,
                   kijun_period: int = 26,
                   senkou_b_period: int = 52) -> Dict[str, pd.Series]:
    """
    Calculate Ichimoku Cloud components.
    """
    high = df['high']
    low = df['low']
    close = df['close']

    def highest_high(series: pd.Series, period: int) -> pd.Series:
        return series.rolling(window=period, min_periods=1).max()

    def lowest_low(series: pd.Series, period: int) -> pd.Series:
        return series.rolling(window=period, min_periods=1).min()

    tenkan_sen = (highest_high(high, tenkan_period) + lowest_low(low, tenkan_period)) / 2
    kijun_sen = (highest_high(high, kijun_period) + lowest_low(low, kijun_period)) / 2
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(kijun_period)
    senkou_span_b = ((highest_high(high, senkou_b_period) + lowest_low(low, senkou_b_period)) / 2).shift(kijun_period)
    chikou_span = close.shift(-kijun_period)

    return {
        'tenkan_sen': tenkan_sen,
        'kijun_sen': kijun_sen,
        'senkou_span_a': senkou_span_a,
        'senkou_span_b': senkou_span_b,
        'chikou_span': chikou_span,
    }


def bollinger_bands(df: pd.DataFrame,
                    window: int = 20,
                    std_multiplier: float = 2.0) -> Dict[str, pd.Series]:
    """
    Calculate Bollinger Bands: upper, middle (SMA), lower.
    """
    if df.empty or len(df) < window:
        return {
            'upper_band': pd.Series(dtype=float),
            'middle_band': pd.Series(dtype=float),
            'lower_band': pd.Series(dtype=float),
        }
    close = df['close']
    middle_band = close.rolling(window=window, min_periods=window).mean()
    std_dev = close.rolling(window=window, min_periods=window).std()
    upper_band = middle_band + std_multiplier * std_dev
    lower_band = middle_band - std_multiplier * std_dev

    return {
        'upper_band': upper_band,
        'middle_band': middle_band,
        'lower_band': lower_band,
    }


def candle_psychology(df: pd.DataFrame) -> pd.Series:
    """
    Recognize candle patterns and return signals:
    1 = Bullish pattern
    -1 = Bearish pattern
    0 = Neutral/no pattern

    Patterns detected:
    - Marubozu (strong candle)
    - Doji
    - Hammer
    - Shooting Star
    - Engulfing (bullish & bearish)
    """
    if df.empty:
        return pd.Series(dtype=int)

    signals = pd.Series(0, index=df.index, dtype=int)

    open_ = df['open']
    close = df['close']
    high = df['high']
    low = df['low']

    body = (close - open_).abs()
    candle_range = high - low
    upper_shadow = high - close.where(close > open_, open_)
    lower_shadow = open_.where(close > open_, close) - low

    # Marubozu: very small/no shadows, long body
    marubozu = (upper_shadow <= 0.05 * body) & (lower_shadow <= 0.05 * body) & (body >= 0.6 * candle_range)
    signals[marubozu & (close > open_)] = 1
    signals[marubozu & (close < open_)] = -1

    # Doji: body very small compared to range
    doji = (body <= 0.1 * candle_range)
    signals[doji] = 0  # Neutral for doji

    # Hammer: small body near high, long lower shadow
    hammer = (body <= 0.3 * candle_range) & (lower_shadow >= 2 * body) & (upper_shadow <= 0.1 * body)
    signals[hammer & (close > open_)] = 1

    # Shooting Star: small body near low, long upper shadow
    shooting_star = (body <= 0.3 * candle_range) & (upper_shadow >= 2 * body) & (lower_shadow <= 0.1 * body)
    signals[shooting_star & (close < open_)] = -1

    # Bullish Engulfing: current candle body engulfs previous bearish candle body
    prev_open = open_.shift(1)
    prev_close = close.shift(1)
    bullish_engulf = (close > open_) & (prev_close < prev_open) & (close >= prev_open) & (open_ <= prev_close)
    signals[bullish_engulf] = 1

    # Bearish Engulfing: current candle body engulfs previous bullish candle body
    bearish_engulf = (close < open_) & (prev_close > prev_open) & (open_ >= prev_close) & (close <= prev_open)
    signals[bearish_engulf] = -1

    return signals.fillna(0).astype(int)


def candle_sentiment(df: pd.DataFrame) -> pd.Series:
    """
    Measures short-term bull/bear pressure based on volume and candle body direction.
    Returns a sentiment score between -1 (bearish) and +1 (bullish).
    """
    if df.empty:
        return pd.Series(dtype=float)

    body = df['close'] - df['open']
    volume = df['volume'].fillna(0)
    sentiment_raw = body * volume
    max_abs = sentiment_raw.abs().max()
    if max_abs == 0:
        return pd.Series(0.0, index=df.index)
    sentiment = sentiment_raw / max_abs
    sentiment = sentiment.clip(-1, 1)
    sentiment.name = "candle_sentiment"
    return sentiment


def trend_confirmation(df: pd.DataFrame,
                       rsi_period: int = 14,
                       alligator_periods: Dict[str, int] = None,
                       ichimoku_periods: Dict[str, int] = None) -> pd.Series:
    """
    Combines RSI, Alligator and Ichimoku to classify trend phases:
    Returns categorical series: 'STRONG_UP', 'STRONG_DOWN', 'SIDEWAYS'
    """
    if alligator_periods is None:
        alligator_periods = {'jaw': 13, 'teeth': 8, 'lips': 5}
    if ichimoku_periods is None:
        ichimoku_periods = {'tenkan': 9, 'kijun': 26, 'senkou_b': 52}

    rsi_series = rsi(df, rsi_period)
    alligator_lines = alligator(df,
                               jaw_period=alligator_periods['jaw'],
                               teeth_period=alligator_periods['teeth'],
                               lips_period=alligator_periods['lips'])
    ichimoku_lines = ichimoku_cloud(df,
                                   tenkan_period=ichimoku_periods['tenkan'],
                                   kijun_period=ichimoku_periods['kijun'],
                                   senkou_b_period=ichimoku_periods['senkou_b'])

    trend = pd.Series(index=df.index, dtype="object")

    # Condition for STRONG_UP:
    # RSI > 60
    # Alligator Lips > Teeth > Jaw (ascending)
    # Price above Senkou Span A and B
    cond_strong_up = (rsi_series > 60) & \
                     (alligator_lines['lips'] > alligator_lines['teeth']) & (alligator_lines['teeth'] > alligator_lines['jaw']) & \
                     (df['close'] > ichimoku_lines['senkou_span_a']) & (df['close'] > ichimoku_lines['senkou_span_b'])

    # Condition for STRONG_DOWN:
    # RSI < 40
    # Alligator Lips < Teeth < Jaw (descending)
    # Price below Senkou Span A and B
    cond_strong_down = (rsi_series < 40) & \
                       (alligator_lines['lips'] < alligator_lines['teeth']) & (alligator_lines['teeth'] < alligator_lines['jaw']) & \
                       (df['close'] < ichimoku_lines['senkou_span_a']) & (df['close'] < ichimoku_lines['senkou_span_b'])

    trend.loc[cond_strong_up] = 'STRONG_UP'
    trend.loc[cond_strong_down] = 'STRONG_DOWN'
    trend.loc[~(cond_strong_up | cond_strong_down)] = 'SIDEWAYS'

    return trend.fillna('SIDEWAYS')


# Exported functions
__all__ = [
    "rsi",
    "alligator",
    "ichimoku_cloud",
    "bollinger_bands",
    "candle_psychology",
    "candle_sentiment",
    "trend_confirmation",
]
