"""
ML Component-Testing model training
"""
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

def test_model_overfitting_small_batch():
    """ML Training Test: Ensure model can overfit on a small batch (capacity check)."""
    X_small = np.random.rand(10, 5)
    y_small = np.array([100, 200, 150, 300, 250, 400, 350, 500, 450, 600])
    
    model = RandomForestRegressor(n_estimators=10, random_state=42, bootstrap=False,  max_depth=None)
    model.fit(X_small, y_small)
    preds = model.predict(X_small)
    
    # Error on training batch should approach zero
    mae = np.mean(np.abs(preds - y_small))
    assert mae < 1.0, f"Model failed small batch overfitting test. MAE: {mae}"

def test_model_training_improves():
    """
    ML Training Test:Check that a stronger/more fully trained model performs better
    than a weak model.

    RandomForest does not expose epoch-wise training loss,
    so we compare prediction error before and after increasing
    model training capacity.
    """
    rng = np.random.RandomState(42)
    X_train = rng.rand(100, 5)

    # Create a learnable relationship
    y_train = (100 * X_train[:, 0] + 50 * X_train[:, 1] + rng.normal(0, 2, 100))

    # Weak model
    weak_model = RandomForestRegressor(n_estimators=1, max_depth=1, random_state=42)
    weak_model.fit(X_train, y_train)
    weak_preds = weak_model.predict(X_train)
    weak_mae = mean_absolute_error(y_train, weak_preds)

    # Stronger model
    trained_model = RandomForestRegressor(n_estimators=100, max_depth=None, random_state=42)
    trained_model.fit(X_train, y_train)
    trained_preds = trained_model.predict(X_train)
    trained_mae = mean_absolute_error(y_train, trained_preds)

    assert trained_mae < weak_mae, (
        f"Training did not improve performance. "
        f"Weak MAE: {weak_mae}, "
        f"Trained MAE: {trained_mae}"
    )
