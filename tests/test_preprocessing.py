import pytest
import numpy as np
from src.data_preprocessing import load_data, preprocess_data
from src.config import TEST_SIZE

def test_load_data():
    X, y = load_data()
    assert X.shape == (150, 4)
    assert y.shape == (150,)
    assert len(np.unique(y)) == 3

def test_preprocess_data():
    X, y = load_data()
    X_train, X_test, _, _ = preprocess_data(X, y)
    assert len(X_test) == int(len(X) * TEST_SIZE)
    assert np.allclose(X_train.mean(axis=0), 0, atol=1e-7)
