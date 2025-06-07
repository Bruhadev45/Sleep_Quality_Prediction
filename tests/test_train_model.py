import os
import train_model


def test_training(tmp_path):
    df = train_model.load_data()
    model, mse = train_model.train_model(df)
    out_path = tmp_path / "model.joblib"
    train_model.save_model(model, out_path)
    assert out_path.exists()
    assert mse >= 0
