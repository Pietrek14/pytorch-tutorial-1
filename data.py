import torch

# Create *known* parameters
weight = 0.7
bias = 0.3

# Create data
start = 0
end = 1
step = 0.02
features = torch.arange(start, end, step).unsqueeze(dim=1)
labels = weight * features + bias

# Use 80% of the data as the training set
train_split = int(0.8 * len(features))

training_features = features[:train_split]
training_labels = labels[:train_split]

testing_features = features[train_split:]
testing_labels = labels[train_split:]
