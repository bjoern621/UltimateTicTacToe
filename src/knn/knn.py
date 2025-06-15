from typing import Any
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


class UTTTDataset(Dataset):
    """Dataset for Ultimate Tic Tac Toe board states"""

    def __init__(self, csv_file, transform=None):
        """
        Args:
            csv_file (string): Path to the CSV file with board data
            transform (callable, optional): Optional transform to apply to the data
        """

        self.data = pd.read_csv(csv_file, sep=";")

        print(f"Dataset loaded with {len(self.data)} samples")

        # Extract features (board cells) and target (winner)
        cell_columns = [f"cell{i}" for i in range(81)]

        # Convert cells to numeric representation: X=1, O=-1, empty=0
        for col in cell_columns:
            self.data[col] = self.data[col].map({"X": 1, "O": -1, " ": 0})

        # Convert forced_board to numeric: None=-1, 0-8 as is
        self.data["forced_board_str"] = self.data["forced_board"].map(
            lambda x: "fb_None" if pd.isna(x) else f"fb_{int(x)}"
        )

        # Create one-hot encoding for forced_board
        forced_board_dummies = pd.get_dummies(self.data["forced_board_str"])

        # Encode the winner: X, O, Draw
        self.label_encoder = LabelEncoder()
        self.data["winner"] = self.label_encoder.fit_transform(self.data["winner"])

        processed_cell_data = self.data[cell_columns]
        features_df = pd.concat([processed_cell_data, forced_board_dummies], axis=1)

        print(
            f"Number of features after one-hot encoding: {features_df.shape[1]}"
        )  # Should be 91 if 10 fb categories

        self.X = features_df.values.astype(np.float32)
        self.y = self.data["winner"].values

        self.transform = transform

        print(f"Classes: {self.label_encoder.classes_}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        features = self.X[idx]
        label = self.y[idx]

        sample = {
            "features": torch.FloatTensor(features),
            "label": torch.tensor(label, dtype=torch.long),
        }

        if self.transform:
            sample = self.transform(sample)

        return sample

    def get_class_names(self):
        """Return the original class names"""
        return self.label_encoder.classes_


class UTTTNeuralNetwork(nn.Module):
    """Neural Network for Ultimate Tic Tac Toe prediction"""

    def __init__(self, input_size=91, hidden_sizes=[128, 64], output_size=3):
        """
        Args:
            input_size (int): Size of input features (81 cells + forced board)
            hidden_sizes (list): Sizes of hidden layers
            output_size (int): Number of output classes (X, O, Draw)
        """
        super(UTTTNeuralNetwork, self).__init__()

        # Build network layers
        layers = []

        # Input layer to first hidden layer
        layers.append(nn.Linear(input_size, hidden_sizes[0]))
        layers.append(nn.ReLU())
        layers.append(nn.BatchNorm1d(hidden_sizes[0]))
        layers.append(nn.Dropout(0.5))

        # Hidden layers
        for i in range(len(hidden_sizes) - 1):
            layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(hidden_sizes[i + 1]))
            layers.append(nn.Dropout(0.5))

        # Output layer
        layers.append(nn.Linear(hidden_sizes[-1], output_size))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


def train_model(
    model: nn.Module,
    train_loader,
    test_loader,
    epochs=10,
    learning_rate=0.001,
    weight_decay=1e-4,
):
    """
    Train the neural network model

    Args:
        model (nn.Module): The neural network model
        train_loader (DataLoader): Training data loader
        test_loader (DataLoader): Testing data loader
        epochs (int): Number of training epochs
        learning_rate (float): Learning rate for optimizer
        weight_decay (float): Weight decay (L2 penalty)

    Returns:
        Trained model and training history
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, "min", patience=3, factor=0.5
    )

    history: dict[str, list[Any]] = {
        "train_loss": [],
        "val_loss": [],
        "train_accuracy": [],
        "val_accuracy": [],
    }

    for epoch in range(epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for i, batch in enumerate(train_loader):
            inputs, labels = batch["features"].to(device), batch["label"].to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            if i % 100 == 99:  # Print every 100 mini-batches
                print(
                    f"[{epoch + 1}, {i + 1}] loss: {running_loss / 100:.3f} accuracy: {100 * correct / total:.2f}%"
                )
                running_loss = 0.0

        # Calculate training metrics
        train_loss, train_acc = evaluate_model(model, train_loader, criterion, device)

        # Validation phase
        val_loss, val_acc = evaluate_model(model, test_loader, criterion, device)

        # Update learning rate
        scheduler.step(val_loss)

        # Store metrics
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_accuracy"].append(train_acc)
        history["val_accuracy"].append(val_acc)

        print(
            f"Epoch {epoch+1}/{epochs} - "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%"
        )

    print("Finished Training")
    return model, history


def evaluate_model(model, data_loader, criterion, device):
    """
    Evaluate the model on the given data loader

    Args:
        model (nn.Module): The neural network model
        data_loader (DataLoader): Data loader for evaluation
        criterion: Loss function
        device: Device to run evaluation on

    Returns:
        Tuple of (average loss, accuracy)
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in data_loader:
            inputs, labels = batch["features"].to(device), batch["label"].to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_loss = running_loss / len(data_loader)
    accuracy = 100 * correct / total

    return avg_loss, accuracy


def predict(model, features, device=None):
    """
    Make a prediction using the trained model

    Args:
        model (nn.Module): Trained neural network model
        features: Input features to make prediction on
        device: Device to run prediction on

    Returns:
        Class prediction and probabilities
    """
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model.eval()

    # Convert to tensor if not already
    if not isinstance(features, torch.Tensor):
        features = torch.FloatTensor(features)

    # Add batch dimension if needed
    if len(features.shape) == 1:
        features = features.unsqueeze(0)

    features = features.to(device)

    with torch.no_grad():
        outputs = model(features)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        _, predicted = torch.max(outputs, 1)

    return predicted.item(), probs.cpu().numpy()


def save_model(model, filepath):
    """Save model to file"""
    torch.save(model.state_dict(), filepath)
    print(f"Model saved to {filepath}")


def load_model(model_class, filepath, **model_params):
    """Load model from file"""
    model = model_class(**model_params)
    model.load_state_dict(torch.load(filepath))
    model.eval()  # Set to evaluation mode
    return model
