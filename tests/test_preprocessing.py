import pytest
import numpy as np
from src.data_preprocessing import load_data, preprocess_data
from src.config import TEST_SIZE

def test_load_data():
    """Test data shape and classes"""
    X, y = load_data()
    assert X.shape == (150, 4)  # 150 samples, 4 features
    assert y.shape == (150,)    # 150 labels
    assert len(np.unique(y)) == 3  # 3 classes

def test_preprocess_data():
    """Test train-test split & scaling"""
    X, y = load_data()
    X_train, X_test, y_train, y_test = preprocess_data(X, y)
    
    # Check split sizes
    assert len(X_test) == int(len(X) * TEST_SIZE)
    
    # Check scaling (mean ~0, std ~1)
    assert np.allclose(X_train.mean(axis=0), 0, atol=1e-7)
    assert np.allclose(X_train.std(axis=0), 1)
