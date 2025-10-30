import tensorflow as tf
import numpy as np
import pandas as pd
from typing import Dict
from logger import logger
from config import Config


class AIModel:
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = None
        self.load_model()

    def load_model(self):
        try:
            self.model = tf.keras.models.load_model(self.model_path)
            logger.info(f"AI model loaded from {self.model_path}")
        except Exception as e:
            logger.error(f"Failed to load AI model from {self.model_path}: {e}")
            self.model = None

    def preprocess(self, df: pd.DataFrame) -> np.ndarray:
        """
        Preprocess OHLCV dataframe to model input tensor.
        Uses last 60 candles.
        Features normalized between 0 and 1.
        """
        if df.empty or len(df) < 60:
            raise ValueError("Insufficient data for AI model preprocessing (need at least 60 candles)")

        df = df.tail(60).copy()

        # Select features: open, high, low, close, volume
        features = df[['open', 'high', 'low', 'close', 'volume']].copy()

        # Normalize features by min-max scaling per feature
        min_vals = features.min()
        max_vals = features.max()
        denom = max_vals - min_vals
        denom[denom == 0] = 1  # prevent division by zero
        norm_features = (features - min_vals) / denom

        # Convert to numpy array shaped (1, 60, 5)
        input_tensor = norm_features.to_numpy().reshape((1, 60, 5)).astype(np.float32)
        return input_tensor

    def predict_candle_direction(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Predict next candle movement direction with confidence score.
        Returns dict: {"prediction": "CALL"/"PUT"/"HOLD", "confidence": float 0-1}
        """
        if self.model is None:
            logger.error("AI model is not loaded.")
            return {"prediction": "HOLD", "confidence": 0.0}

        try:
            input_tensor = self.preprocess(df)
            preds = self.model.predict(input_tensor, verbose=0)
            # Assuming model output shape (1, 3) for [CALL, PUT, HOLD]
            if preds.shape[-1] == 3:
                idx = np.argmax(preds[0])
                confidence = float(np.max(preds[0]))
                prediction_map = {0: "CALL", 1: "PUT", 2: "HOLD"}
                prediction = prediction_map.get(idx, "HOLD")
                return {"prediction": prediction, "confidence": confidence}
            else:
                logger.error(f"Unexpected AI model output shape: {preds.shape}")
                return {"prediction": "HOLD", "confidence": 0.0}
        except Exception as e:
            logger.error(f"AI model prediction error: {e}")
            return {"prediction": "HOLD", "confidence": 0.0}


ai_model = AIModel(Config.MODEL_PATH)
