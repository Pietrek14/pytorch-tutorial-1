import matplotlib.pyplot as plt
import torch

from data import *

# Helper function for plotting the data


def plot_data(
        train_data=training_features,
        train_labels=training_labels,
        test_data=testing_features,
        test_labels=testing_labels,
        predictions=None):

    plt.figure(figsize=(10, 7))

    # Plot training data in blue
    plt.scatter(train_data, train_labels, c="b", s=4, label="Training data")

    # Plot test data in green
    plt.scatter(test_data, test_labels, c="g", s=4, label="Testing data")

    if predictions is not None:
        # Plot the predictions in red (predictions were made on the test data)
        plt.scatter(test_data, predictions, c="r", s=4, label="Predictions")

    # Show the legend
    plt.legend(prop={"size": 14})

    plt.show()


def show_model_performance(model):
    # Put the model in the evaluation mode
    model.eval()

    # Inference mode makes the forward-passes faster
    with torch.inference_mode():
        preds = model(testing_features)
        plot_data(predictions=preds)


def plot_loss_curves(epoch_count, train_loss_values, test_loss_values):
    plt.plot(epoch_count, train_loss_values, label="Train loss")
    plt.plot(epoch_count, test_loss_values, label="Test loss")
    plt.title("Training and test loss curves")
    plt.ylabel("Loss")
    plt.xlabel("Epochs")
    plt.legend()
    plt.show()
