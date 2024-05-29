import numpy as np
from sklearn.linear_model import LogisticRegression as LR
from joblib import dump, load
import os
from sklearn.preprocessing import StandardScaler

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
    # dataset_cleaned['ShapeFactor5'] = dataset_cleaned['MajorAxisLength'] / dataset_cleaned['Perimeter']
    # dataset_cleaned['ShapeFactor6'] = dataset_cleaned['MinorAxisLength'] / dataset_cleaned['Perimeter']
    # dataset_cleaned['ShapeFactor7'] = dataset_cleaned['Eccentricity'] * dataset_cleaned['Area']
    # dataset_cleaned['ShapeFactor8'] = dataset_cleaned['Eccentricity'] * dataset_cleaned['Perimeter']
    # dataset_cleaned['ShapeFactor9'] = dataset_cleaned['Extent'] * dataset_cleaned['Area']
    # dataset_cleaned['FormFactor'] = (4 * np.pi * dataset_cleaned['Area']) / (dataset_cleaned['Perimeter'] ** 2)
    # dataset_cleaned['Elongation'] = (dataset_cleaned['MajorAxisLength'] - dataset_cleaned['MinorAxisLength']) / (dataset_cleaned['MajorAxisLength'] + dataset_cleaned['MinorAxisLength'])
    feature_values = np.array([get_value(k, 0.0) for k in features])
    feature_values = np.append(feature_values, (feature_values[2] / feature_values[1]))
    feature_values = np.append(feature_values, (feature_values[3] / feature_values[1]))
    feature_values = np.append(feature_values, (feature_values[5] * feature_values[0]))
    feature_values = np.append(feature_values, (feature_values[5] * feature_values[1]))
    feature_values = np.append(feature_values, (feature_values[8] * feature_values[0]))
    feature_values = np.append(feature_values, ((4 * np.pi * feature_values[0]) / (feature_values[1] ** 2)))
    feature_values = np.append(feature_values, ((feature_values[2] - feature_values[3]) / (feature_values[2] + feature_values[3])))
    scaler = StandardScaler()
    feature_values = scaler.fit_transform(feature_values.reshape(1, -1))
    return feature_values


# friendly version
def get_prediction(model, input_dict):
    inp = prepare_input(input_dict)
    classes = ['SEKER', 'BARBUNYA', 'BOMBAY', 'CALI', 'HOROZ', 'SIRA', 'DERMASON']
    return classes[model.predict(inp)[0]]


if __name__ == "__main__":
    print("Loading model")
    model = load_model("model.joblib")
else:
    print("Loading model")
    model = load_model("model.joblib")
