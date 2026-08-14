"""
ML Component-Testing model inference
"""
import numpy as np
from sklearn.ensemble import RandomForestRegressor

def test_inference_output_range():
    """ML Inference Test: Verify predictions lie within realistic boundaries."""
    X_test = np.random.rand(5, 15)
    # Mocking trained model with standard range
    model = RandomForestRegressor(n_estimators=10, random_state=42)
    model.fit(np.random.rand(20, 15), np.random.uniform(50000, 500000, 20))
    
    preds = model.predict(X_test)
    assert len(preds) == 5, "Output shape mismatch."
    assert np.all(preds >= 0), "Inference test failed: Negative prices generated!"
    assert np.all(preds <= 1000000), "Inference test failed: Unrealistic upper price bound!"

def test_inference_invariance():
    """
    ML Inference Test: Verify that identical copies of the same input produce identical 
    predictions.
    """
    rng = np.random.RandomState(42)
    X_train = rng.rand(50, 5)
    y_train = (100 * X_train[:, 0] + 50 * X_train[:, 1])

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    X_sample = rng.rand(5, 5)
    prediction_original = model.predict(X_sample)

    # Create an identical copy
    X_copy = X_sample.copy()
    prediction_copy = model.predict(X_copy)

    assert np.allclose(prediction_original,prediction_copy), (
        "Predictions changed for identical input data."
    )
