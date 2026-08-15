"""
Modular Model Building, Evaluation, Persistence & Visualization
"""

import json
import os
from typing import Any, Dict, List, Optional, Tuple

import joblib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from app.logger import logger
from utils import get_short_path

matplotlib.use("Agg")  # Non-interactive backend for headless server execution

class ModelTrainer:
    """Handles training, evaluating, persisting regression models, and generating report figures."""

    def __init__(self, models_dir: str = "models", figures_dir: str = "figures"):
        self.models_dir = os.path.abspath(models_dir)
        self.figures_dir = os.path.abspath(figures_dir)

        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.figures_dir, exist_ok=True)

        logger.info(
            "Initialized ModelTrainer - Models Dir: %s", get_short_path(self.models_dir)
        )
        logger.info(
            "Initialized ModelTrainer - Figures Dir: %s", get_short_path(self.figures_dir)
        )

        self.models: Dict[str, Any] = {}
        self.metrics: Dict[str, Dict[str, Any]] = {}
        self.predictions: Dict[str, np.ndarray] = {}  # Stores evaluation predictions

    def train_linear_regression(
        self, X_train: np.ndarray, y_train: pd.Series # pylint: disable=invalid-name
    ) -> LinearRegression:
        """
        Train a Linear Regression model using the provided training data.

        Args:
            X_train: Training feature matrix.
            y_train: Training target values.

        Returns:
            The trained LinearRegression model.

        Raises:
            Exception: If model training fails.
        """
        logger.info("Training Linear Regression model...")
        try:
            model = LinearRegression()
            model.fit(X_train, y_train) # pylint: disable=invalid-name
            self.models["linear_regression"] = model
            logger.info("Linear Regression training completed successfully.")
            return model
        except Exception as e:
            logger.error("Failed to train Linear Regression model: %s", str(e))
            raise

    def train_random_forest(
        self,
        X_train: np.ndarray, # pylint: disable=invalid-name
        y_train: pd.Series,
        n_estimators: int = 200,
        max_depth: int = 12,
        random_state: int = 42,
    ) -> RandomForestRegressor:
        """
        Train a Random Forest Regressor using the provided training data.

        Args:
            X_train: Training feature matrix.
            y_train: Training target values.
            n_estimators: Number of trees in the forest.
            max_depth: Maximum depth of each decision tree.
            random_state: Seed used to ensure reproducible results.

        Returns:
            The trained RandomForestRegressor model.

        Raises:
            Exception: If model training fails.
        """
        logger.info("Training Random Forest Regressor model...")
        try:
            model = RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=random_state,
                n_jobs=-1,
            )
            model.fit(X_train, y_train) # pylint: disable=invalid-name
            self.models["random_forest"] = model
            logger.info("Random Forest training completed successfully.")
            return model
        except Exception as e:
            logger.error("Failed to train Random Forest model: %s", str(e))
            raise

    def evaluate_model(
        self, model_name: str,
        X_test: np.ndarray, # pylint: disable=invalid-name
        y_test: pd.Series
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Evaluate a trained model using test data.

        Generates predictions and calculates Mean Absolute Error (MAE),
        Root Mean Squared Error (RMSE), and R-squared (R2). The predictions
        and evaluation metrics are stored for later use.

        Args:
            model_name: Name of the trained model to evaluate.
            X_test: Test feature matrix.
            y_test: Actual target values for the test dataset.

        Returns:
            A tuple containing the model predictions and evaluation metrics.

        Raises:
            ValueError: If the specified model has not been trained.
        """

        if model_name not in self.models:
            logger.error("Model %s has not been trained yet.", model_name)
            raise ValueError(f"Model '{model_name}' is not available in trainer.")

        logger.info("Evaluating model performance: %s", model_name)
        model = self.models[model_name]
        predictions = model.predict(X_test)

        # Save predictions inside class state for visualization methods
        self.predictions[model_name] = predictions

        mae = float(mean_absolute_error(y_test, predictions))
        rmse = float(np.sqrt(mean_squared_error(y_test, predictions)))
        r2 = float(r2_score(y_test, predictions))

        metrics_dict = {
            "model": model_name,
            "MAE": round(mae, 4),
            "RMSE": round(rmse, 4),
            "R2": round(r2, 4),
        }

        self.metrics[model_name] = metrics_dict
        logger.info("[%s] -> MAE: %.2f | RMSE: %.2f | R2: %.4f", model_name, mae, rmse, r2)

        return predictions, metrics_dict

    def save_models_and_metrics(self) -> None:
        """
        Save all trained models and their evaluation metrics to disk.

        Each trained model is serialized as a Joblib file in the configured
        models directory. Available evaluation metrics are saved as a JSON file.

        Raises:
            Exception: If a model or metrics file cannot be saved.
        """
        if not self.models:
            logger.warning("No models found to dump/persist.")
            return

        logger.info(
            "Saving trained models to directory: %s",
            get_short_path(self.models_dir)
        )
        for name, model in self.models.items():
            file_path = os.path.join(self.models_dir, f"{name}.joblib")
            try:
                joblib.dump(model, file_path)
                logger.info("Dumped model artifact: %s", get_short_path(file_path))
            except Exception as e:
                logger.error("Failed to save model '%s': %s", name, str(e))
                raise

        if self.metrics:
            metrics_path = os.path.join(self.models_dir, "metrics.json")
            try:
                metrics_list = list(self.metrics.values())
                with open(metrics_path, "w", encoding="utf-8") as f:
                    json.dump(metrics_list, f, indent=2)
                logger.info(
                    "Saved evaluation metrics report to: %s", get_short_path(metrics_path)
                )
            except Exception as e:
                logger.error("Failed to write metrics report: %s", str(e))
                raise

    def generate_visualizations(
        self,
        df_cleaned: pd.DataFrame,
        feature_cols: List[str],
        y_test: pd.Series,
        rf_preds: Optional[np.ndarray] = None,
        target_col: str = "MedianHouseValue",
    ) -> None:
        """
        VISUALIZATION (evidence / screenshots for report)
        Generates and saves performance evaluation plots and data distributions.
        """

        logger.info("Generating visualization figures...")

        try:
            plt.style.use("seaborn-v0_8-whitegrid")

            # Fallback to stored evaluation predictions if not passed explicitly
            if rf_preds is None:
                rf_preds = self.predictions.get("random_forest")

            # (a) Actual vs Predicted - Random Forest
            if rf_preds is not None:
                plt.figure(figsize=(6, 6))
                plt.scatter(y_test, rf_preds, alpha=0.3, s=12, color="#2E86AB")
                plt.plot(
                    [y_test.min(), y_test.max()],
                    [y_test.min(), y_test.max()],
                    "r--",
                    lw=2,
                )
                plt.xlabel("Actual Median House Value (US$)")
                plt.ylabel("Predicted Median House Value (US$)")
                plt.title("Random Forest: Actual vs Predicted House Prices")
                plt.tight_layout()
                rf_actual_vs_pred_path = os.path.join(
                    self.figures_dir, "actual_vs_predicted_rf.png"
                )
                plt.savefig(rf_actual_vs_pred_path, dpi=150)
                plt.close()
                logger.info("Saved plot: %s", get_short_path(rf_actual_vs_pred_path))

            # (b) Feature importance
            if "random_forest" in self.models:
                rf_model = self.models["random_forest"]
                importances = pd.Series(
                    rf_model.feature_importances_, index=feature_cols
                )
                importances = importances.sort_values(ascending=True)

                csv_importance_path = os.path.join(
                    self.figures_dir, "feature_importance.csv"
                )
                importances.to_csv(csv_importance_path)

                plt.figure(figsize=(7, 5))
                importances.plot(kind="barh", color="#2E86AB")
                plt.title("Random Forest Feature Importance")
                plt.xlabel("Importance")
                plt.tight_layout()
                img_importance_path = os.path.join(
                    self.figures_dir, "feature_importance.png"
                )
                plt.savefig(img_importance_path, dpi=150)
                plt.close()
                logger.info(
                    "Saved feature importances: %s", get_short_path(img_importance_path)
                )

            # (c) Model comparison bar chart
            if self.metrics:
                metrics_df = pd.DataFrame(list(self.metrics.values())).set_index(
                    "model"
                )
                plt.figure(figsize=(6, 4))
                metrics_df[["MAE", "RMSE"]].plot(
                    kind="bar", color=["#2E86AB", "#F26419"]
                )
                plt.title("Model Comparison: MAE & RMSE (lower is better)")
                plt.ylabel("Error (US$)")
                plt.xticks(rotation=0)
                plt.tight_layout()
                comp_path = os.path.join(self.figures_dir, "model_comparison.png")
                plt.savefig(comp_path, dpi=150)
                plt.close()
                logger.info(
                    "Saved model comparison chart: %s", get_short_path(comp_path)
                )

            # (d) Target distribution after cleaning
            if target_col in df_cleaned.columns:
                plt.figure(figsize=(6, 4))
                df_cleaned[target_col].hist(bins=40, color="#2E86AB")
                plt.title("Distribution of Median House Value (post-cleaning)")
                plt.xlabel("Median House Value (US$)")
                plt.ylabel("Frequency")
                plt.tight_layout()
                dist_path = os.path.join(self.figures_dir, "target_distribution.png")
                plt.savefig(dist_path, dpi=150)
                plt.close()
                logger.info(
                    "Saved target distribution chart: %s", get_short_path(dist_path)
                )

            logger.info(
                "All figures successfully saved to: %s", get_short_path(self.figures_dir)
            )

        except Exception as e:
            logger.error("Failed to generate visualization figures: %s", str(e))
            raise
