""" Module to run the ml classification model """

import os
import torch
from torchvision import transforms
from model import *

# import the pre-trained model
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, "mnist-cnn.pth")

model = CNNModel()
model.load_state_dict(
    torch.load(model_path, map_location=torch.device("cpu"), weights_only=True)
)
model.eval()

test_transform = transforms.Compose(
    [
        transforms.ToTensor(),
        # inverted here due to nature of how data is outputted from webpage
        InvertGrayscale(),
        transforms.Resize((28, 28)),
        transforms.Normalize((254.8692,), (0.3015,)),
    ]
)


def mnist_classify(data):
    """Receives web-app data and returns the model output"""
    model_input = test_transform(data).unsqueeze(0)
    with torch.no_grad():
        output = model(model_input)
        prediction = output.argmax(dim=1, keepdim=True).item()
    return prediction
