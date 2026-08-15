"""
Data Preparation & Feature Engineering
"""

import os

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from app.logger import logger
from utils import get_short_path


class DataPreparation:
    """
    Handle data cleaning, feature engineering, and feature scaling.

    This class prepares raw housing data for machine learning by cleaning
    missing or duplicate values, creating derived features, and applying
    feature scaling using StandardScaler.
    """

    def __init__(self, models_dir: str = "models"):
        """
        Initialize the data preparation pipeline.

        Args:
            models_dir: Directory where the fitted scaler artifact is stored.
        """
        self.scaler = StandardScaler()
        self.feature_cols = []
        self.models_dir = os.path.abspath(models_dir)
        os.makedirs(self.models_dir, exist_ok=True)

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and preprocess the input dataset.

        Removes duplicate records, imputes missing total_bedrooms values,
        clips extreme target outliers, and one-hot encodes the
        ocean_proximity categorical feature.

        Args:
            df: Raw input dataset.

        Returns:
            The cleaned and preprocessed dataset.
        """
        logger.info("Starting data cleaning phase...")
        df = df.drop_duplicates()

        # Log missing value report
        missing_value = df.isnull().sum()
        logger.info("Missing Values: %s", missing_value)

        # Impute missing total_bedrooms with median (only column with nulls)
        if "total_bedrooms" in df.columns:
            median_val = df["total_bedrooms"].median()
            df["total_bedrooms"] = df["total_bedrooms"].fillna(median_val)
            logger.info("Imputed missing total_bedrooms with median: %s", median_val)

        # Cap extreme outliers in target (winsorize at 1st/99th percentile)
        if "MedianHouseValue" in df.columns:
            low, high = df["MedianHouseValue"].quantile([0.01, 0.99])
            df["MedianHouseValue"] = df["MedianHouseValue"].clip(low, high)
            logger.info("Clipped MedianHouseValue outliers between [%.2f, %.2f]", low, high)

        # One-hot encoding the categorical feature
        if "ocean_proximity" in df.columns:
            df = pd.get_dummies(df, columns=["ocean_proximity"], drop_first=True)
            logger.info("One-hot encoded 'ocean_proximity' column.")

        return df

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create additional features from the input dataset.

        Generates RoomsPerHousehold, BedroomsPerRoom, and
        PopulationPerHousehold features. Any missing numeric values created
        during feature engineering are filled using the median.

        Args:
            df: Cleaned dataset used for feature engineering.

        Returns:
            The dataset with additional engineered features.
        """
        logger.info("Executing feature engineering...")
        df["RoomsPerHousehold"] = df["total_rooms"] / df["households"].replace(
            0, np.nan
        )
        df["BedroomsPerRoom"] = df["total_bedrooms"] / df["total_rooms"].replace(
            0, np.nan
        )
        df["PopulationPerHousehold"] = df["population"] / df["households"].replace(
            0, np.nan
        )
        df = df.fillna(df.median(numeric_only=True))
        return df

    def fit_transform_scaler(
        self,
        X_train: pd.DataFrame # pylint: disable=invalid-name
    ) -> np.ndarray:
        """
        Fit the StandardScaler on training data and transform the features.

        The fitted scaler is saved as a Joblib artifact for use during
        inference.

        Args:
            X_train: Training feature dataset.

        Returns:
            The scaled training feature array.

        Raises:
            Exception: If the scaler artifact cannot be saved.
        """
        logger.info("Fitting and applying StandardScaler on training dataset...")
        X_train_scaled = self.scaler.fit_transform(X_train) # pylint: disable=invalid-name
        self.feature_cols = X_train.columns.tolist() # pylint: disable=invalid-name

        # Save scaler immediately after fitting
        scaler_path = os.path.join(self.models_dir, "scaler.joblib")
        try:
            joblib.dump(self.scaler, scaler_path)
            logger.info(
                "Scaler artifact successfully dumped to: %s",
                get_short_path(scaler_path)
            )
        except Exception as e:
            logger.error("Failed to dump scaler artifact: %s", str(e))
            raise

        return X_train_scaled

    def transform_scaler(
            self,
            X_test: pd.DataFrame # pylint: disable=invalid-name
    ) -> np.ndarray:
        """
        Transform test features using the previously fitted scaler.

        Args:
            X_test: Test feature dataset.

        Returns:
            The scaled test feature array.
        """
        return self.scaler.transform(X_test)
