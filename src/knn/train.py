import sys

import torch
from knn import UTTTDataset, train_model, evaluate_model, save_model, UTTTNeuralNetwork
from torch.utils.data import DataLoader
import torch.nn as nn


def main():
    try:
        filename = sys.argv[1]
        dataset = UTTTDataset(filename)

        # Split dataset into train and test sets
        train_size = int(0.8 * len(dataset))
        test_size = len(dataset) - train_size
        train_dataset, test_dataset = torch.utils.data.random_split(
            dataset, [train_size, test_size]
        )

        print(
            f"Training on {len(train_dataset)} samples, testing on {len(test_dataset)} samples"
        )

        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

        # Create model
        model = UTTTNeuralNetwork(
            input_size=91
        )  # 81 cells + 10 forced_board (one-hot encoded)

        # Train model
        trained_model, history = train_model(
            model,
            train_loader,
            test_loader,
            epochs=20,
            learning_rate=0.001,  # Wie schnell er seine Gewichte anpasst
            weight_decay=1e-3,  # L2 regularization
        )

        # Save model
        save_model(trained_model, "uttt_model.pth")

        # Evaluate on test set
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        criterion = nn.CrossEntropyLoss()
        test_loss, test_acc = evaluate_model(
            trained_model, test_loader, criterion, device
        )
        print(f"Test accuracy: {test_acc:.2f}%")

    except Exception as e:
        print(f"Error during training: {str(e)}")


if __name__ == "__main__":
    main()
