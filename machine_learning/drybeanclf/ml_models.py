import numpy as np
from sklearn.linear_model import LogisticRegression as LR
from joblib import dump, load
import os


# def train(X, y, model_output='diabetes.joblib'):
#     print("Training")
#     model = LR(max_iter=1000)
#     model.fit(X, y)
#     print(f"Save model to {model_output}")
#     dump(model, model_output)


def load_model(model_path):
    assert os.path.exists(model_path)
    return load(model_path)


def prepare_input(input_dict):
    def get_value(key, default_value):
        return float(input_dict[key]) if key in input_dict else default_value

    features = [
        "Area",
        "Perimeter",
        "MajorAxisLength",
        "MinorAxisLength",
        "AspectRation",
        "Eccentricity",
        "ConvexArea",
        "EquivDiameter",
        "Extent",
        "Solidity",
        "roundness",
        "Compactness",
        "ShapeFactor1",
        "ShapeFactor2",
        "ShapeFactor3",
        "ShapeFactor4",
    ]
    feature_values = np.array([get_value(k, 0.0) for k in features])
    return feature_values.reshape(1, -1)


# friendly version
def get_prediction(model, input_dict):
    inp = prepare_input(input_dict)
    return model.predict(inp)[0]


if __name__ == "__main__":
    print("Loading model")
    model = load_model("model.pkl")
else:
    print("Loading model")
    model = load_model("model.pkl")
