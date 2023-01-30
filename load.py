import torch

from model import LinearRegressionModel
from util import show_model_performance

model = LinearRegressionModel()

model.load_state_dict(torch.load(f="linear_model_1.pth"))

show_model_performance(model)
