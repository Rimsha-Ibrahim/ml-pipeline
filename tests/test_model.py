from src.train_model import train_and_save_model

def test_model_accuracy():
    accuracy = train_and_save_model()
    assert accuracy > 0.8
