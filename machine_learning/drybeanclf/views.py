from django.shortcuts import render
from .ml_models import model, get_prediction


def index(request):
    context = {"status": "Input test results"}
    return render(request, "drybeans/index.html", context)


def predict(request):
    context = {"status": "Input test results"}
    if request.method == "POST":
        prediction = get_prediction(model, request.POST)
        context = {
            "status": f"Prediction = {prediction}"
        }
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
        for key in features:
            if key in request.POST:
                context[key] = request.POST[key]
    return render(request, "drybeans/index.html", context)
