from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
from .config import MODEL_PATH, RANDOM_STATE
from .data_preprocessing import load_data, preprocess_data

def train_and_save_model():
    """Train and save the model"""
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
