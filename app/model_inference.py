"""
Model Loading & Real-time Inference
"""

import joblib

from app.logger import logger
from utils import get_short_path

# pylint: disable=too-few-public-methods
class PricePredictor:
    """
    Load trained model artifacts and perform real-time predictions.

    This class loads a trained machine learning model and its associated
    feature scaler, then uses them to transform input features and generate
    house price predictions.
    """

    def __init__(self, model_path: str, scaler_path: str):
        self.model = self._load_artifact(model_path, "Model")
        self.scaler = self._load_artifact(scaler_path, "Scaler")

    def _load_artifact(self, path: str, artifact_type: str):
        """
        Load and deserialize a machine learning artifact from disk.

        Args:
            path: File path of the serialized artifact.
            artifact_type: Descriptive name of the artifact being loaded.

        Returns:
            The deserialized artifact.

        Raises:
            FileNotFoundError: If the artifact file does not exist.
            RuntimeError: If the artifact cannot be deserialized.
        """

        logger.info("Loading %s artifact from: %s", artifact_type, get_short_path(path))

        try:
            artifact = joblib.load(path)
            logger.info("%s artifact loaded successfully.", artifact_type)
            return artifact
        except FileNotFoundError:
            logger.error("Critical error: %s file not found at %s", artifact_type, path)
            raise
        except Exception as e:
            logger.error("Failed to deserialize %s: %s", artifact_type, str(e))
            raise RuntimeError("Artifact loading error: %s", str(e)) from e

    def predict(self, feature_df):
        """
        Generate predictions from the provided input features.

        The input features are first transformed using the loaded scaler and
        then passed to the trained machine learning model.

        Args:
            feature_df: Input feature data used for prediction.

        Returns:
            An array containing the predicted house prices.

        Raises:
            ValueError: If feature transformation or model inference fails.
        """

        try:
            scaled_features = self.scaler.transform(feature_df)
            predictions = self.model.predict(scaled_features)
            logger.info("Generated %s predictions successfully.", len(predictions))

            return predictions

        except Exception as e:
            logger.error("Inference pipeline execution error: %s", str(e))
            raise ValueError(f"Inference error: {str(e)}") from e
