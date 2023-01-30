# https://www.learnpytorch.io/01_pytorch_workflow/

import torch
from torch import nn

from model import LinearRegressionModel
from data import *
from util import *

# RNG seed
torch.manual_seed(42)

# Init model
model = LinearRegressionModel()

# Set GPU (cuda) if available
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

model.to(device)

# The model is currently random, so it will produce incorrect predictions
show_model_performance(model)

# Mean Absolute Error (MAE), calculating the error for every value and taking its average
loss_fn = nn.L1Loss()

optimizer = torch.optim.SGD(params=model.parameters(),  # Parameters to optimize
                            lr=0.0025)  # Learning rate

# Optimization loop

epochs = 2000

# Create empty loss lists to track values
train_loss_values = []
test_loss_values = []
epoch_count = []

for epoch in range(epochs):
    # Training

    # Put model in training mode (this is the default state of a model)
    model.train()

    # 1. Forward pass on train data using the forward() method inside
    preds = model(training_features)

    # 2. Calculate the loss (how different are our models predictions to the ground truth)
    loss = loss_fn(preds, training_labels)

    # 3. Zero grad of the optimizer
    optimizer.zero_grad()

    # 4. Loss backwards
    loss.backward()

    # 5. Progress the optimizer
    optimizer.step()

    # Testing

    # Put the model in evaluation mode
    model.eval()

    with torch.inference_mode():
        # 1. Forward pass on test data
        test_pred = model(testing_features)

        # 2. Caculate loss on test data
        # predictions come in torch.float datatype, so comparisons need to be done with tensors of the same type
        test_loss = loss_fn(test_pred, testing_labels.type(torch.float))

        # Print out what's happening
        if epoch % 100 == 0:
            epoch_count.append(epoch)
            train_loss_values.append(loss.detach().numpy())
            test_loss_values.append(test_loss.detach().numpy())
            print(
                f"Epoch: {epoch} | MAE Train Loss: {loss} | MAE Test Loss: {test_loss} ")

# Plot the loss curves
plot_loss_curves(epoch_count, train_loss_values, test_loss_values)

# Find our model's learned parameters
print("The model learned the following values for weights and bias:")
print(model.state_dict())
print("\nAnd the original values for weights and bias are:")
print(f"weights: {weight}, bias: {bias}")

# The results should be about correct now
show_model_performance(model)

# Save the model to a file
torch.save(obj=model.state_dict(), f="linear_model_1.pth")
