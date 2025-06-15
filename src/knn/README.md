# Ultimate Tic Tac Toe Neural Network

This module implements a neural network (KNN - Künstliches Neuronales Netz) for predicting the outcome of Ultimate Tic Tac Toe games.

## Features

-   Neural network to predict game outcomes (X wins, O wins, or Draw)
-   Takes the full board state (81 cells) plus the forced board as input
-   Data preprocessing and transformation
-   Training and evaluation pipeline

## Installation

To use this neural network, you'll need to install the following dependencies:

```bash
pip install torch torchvision pandas scikit-learn numpy
```

## Usage

### Training the model

```python
from knn import UTTTDataset, UTTTNeuralNetwork, train_model, save_model

# Prepare dataset
dataset = UTTTDataset("path/to/boards_dataset.csv")
train_dataset, test_dataset = torch.utils.data.random_split(
    dataset, [int(0.8 * len(dataset)), len(dataset) - int(0.8 * len(dataset))]
)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Create and train model
model = UTTTNeuralNetwork(input_size=82)  # 81 cells + 1 forced_board
trained_model, history = train_model(model, train_loader, test_loader, epochs=20)

# Save the trained model
save_model(trained_model, "uttt_model.pth")
```

### Making predictions

```python
from knn import UTTTNeuralNetwork, predict, load_model

# Load a trained model
model = load_model(UTTTNeuralNetwork, "uttt_model.pth", input_size=82)

# Prepare input features
# features should be a tensor of shape (82,) containing:
# - 81 values for each cell (1 for X, -1 for O, 0 for empty)
# - 1 value for forced board (0-8 or -1 for None)
features = torch.tensor([...])  # Your board state here

# Get prediction
predicted_class, probabilities = predict(model, features)
print(f"Predicted outcome: {predicted_class}")
print(f"Probabilities: {probabilities}")
```

## Data Format

The neural network expects a CSV file with the following columns:

-   `cell0` through `cell80`: The 81 cells of the 9 small boards (X, O, or empty)
-   `forced_board`: The index of the forced board (0-8 or None)
-   `winner`: The outcome of the game (X, O, or Draw)

## Model Architecture

The neural network consists of:

-   Input layer (82 neurons)
-   Hidden layers with batch normalization and dropout
-   Output layer (3 neurons for X, O, Draw)

## Performance

On the test dataset, the model achieves approximately X% accuracy in predicting game outcomes.
