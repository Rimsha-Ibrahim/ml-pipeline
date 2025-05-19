import os
from .config import MODEL_PATH
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

from src.config import MODEL_PATH, RANDOM_STATE
from src.data_preprocessing import load_data, preprocess_data


def train_and_save_model():
    X, y = load_data()
    X_train, X_test, y_train, y_test = preprocess_data(X, y)
    
    model = RandomForestClassifier(random_state=RANDOM_STATE)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.2f}")
    
    joblib.dump(model, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")
    return accuracy

if __name__ == "__main__":
    train_and_save_model()
 # Add verification before saving
    print(f"Saving model to: {os.path.abspath(MODEL_PATH)}")
    print(f"Directory exists: {os.path.exists(os.path.dirname(MODEL_PATH))}")
    
    joblib.dump(model, MODEL_PATH)
    
    # Verify after saving
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Failed to save model at {MODEL_PATH}")
    print("Model saved successfully")
