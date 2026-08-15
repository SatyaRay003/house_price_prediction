"""
Root Orchestration Script for Data Prep & Model Training
"""

import os
from sklearn.model_selection import train_test_split
from app.data_ingestion import DataIngestion
from app.feature_engineering import DataPreparation
from app.model_training import ModelTrainer
from app.logger import logger


def run_pipeline():
    logger.info("================ Starting ML Training Pipeline ================")

    # 1. Resolve Paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, "data", "housing.csv")
    models_dir = os.path.join(base_dir, "models")

    # 2. Data Ingestion
    df_raw = DataIngestion.load_dataset(data_path)

    # 3. Data Cleaning & Feature Engineering
    data_prep = DataPreparation()
    df_cleaned = data_prep.clean_data(df_raw)
    df_features = data_prep.engineer_features(df_cleaned)

    # 4. Train / Test Split & Feature Scaling
    target_col = "MedianHouseValue"
    X = df_features.drop(columns=[target_col])
    y = df_features[target_col]


    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    X_train_scaled = data_prep.fit_transform_scaler(X_train)
    X_test_scaled = data_prep.transform_scaler(X_test)

    # 5. Model Training, Evaluation & Artifact Persistence
    trainer = ModelTrainer(models_dir=models_dir)

    # Train Linear Regression & Random Forest
    trainer.train_linear_regression(X_train_scaled, y_train)
    trainer.train_random_forest(X_train_scaled, y_train)

    # Evaluate
    trainer.evaluate_model("linear_regression", X_test_scaled, y_test)
    trainer.evaluate_model("random_forest", X_test_scaled, y_test)

    # Save trained artifacts (.joblib & metrics.json)
    trainer.save_models_and_metrics()

    # Generate the visualizations
    trainer.generate_visualizations(
        df_cleaned=df_cleaned,
        feature_cols=X.columns.tolist(),
        y_test=y_test
    )

    logger.info(
        "================ Training Pipeline Completed Successfully ================"
    )


if __name__ == "__main__":
    run_pipeline()