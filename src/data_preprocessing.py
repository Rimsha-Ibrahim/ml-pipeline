from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
from .config import RANDOM_STATE, TEST_SIZE

def load_data():
    """Load Iris dataset"""
    iris = load_iris()
    return iris.data, iris.target

def preprocess_data(X, y):
    """Split & scale data"""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test
