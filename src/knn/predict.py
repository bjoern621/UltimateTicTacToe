import sys
import torch
import pandas as pd
import os

from knn import UTTTDataset, UTTTNeuralNetwork, load_model, predict


def main():
    if len(sys.argv) != 3:
        print("Usage: python predict.py <dataset_file> <model_file>")
        sys.exit(1)

    filename = sys.argv[1]
    model_path = sys.argv[2]

    try:
        # Load dataset
        print(f"Loading dataset from {filename}...")
        dataset = UTTTDataset(filename)
        print(f"Dataset loaded successfully with {len(dataset)} samples")

        # Load model
        print(f"Loading model from {model_path}...")
        trained_model = load_model(UTTTNeuralNetwork, model_path, input_size=91)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"Model loaded successfully (using {device})")

        # Calculate statistics on the whole dataset
        print("\nCalculating model statistics...")
        all_true_labels = []
        all_predictions = []
        class_names = dataset.get_class_names()

        total_samples = len(dataset)

        for i in range(total_samples):
            sample = dataset[i]
            features = sample["features"]
            true_label = sample["label"].item()

            pred_class, _ = predict(trained_model, features, device)

            all_true_labels.append(true_label)
            all_predictions.append(pred_class)

            # Show simple progress indicator
            if (i + 1) % 20 == 0:
                print(f"Processed {i+1}/{total_samples} samples")

        # Calculate overall accuracy
        correct_predictions = sum(
            1 for true, pred in zip(all_true_labels, all_predictions) if true == pred
        )
        accuracy = correct_predictions / total_samples
        print(
            f"\nOverall accuracy on {total_samples} samples: {accuracy:.4f} ({int(accuracy * 100)}%)"
        )

        # Create a simple confusion matrix
        print("\nConfusion Matrix:")

        # Print header
        print(f"{'Actual\\Pred':^10}", end="")
        for class_name in class_names:
            print(f"{class_name:^10}", end="")
        print()

        # Build and print confusion matrix
        confusion_matrix = {}
        for true_class in range(len(class_names)):
            confusion_matrix[true_class] = {}
            for pred_class in range(len(class_names)):
                confusion_matrix[true_class][pred_class] = 0

        for true, pred in zip(all_true_labels, all_predictions):
            confusion_matrix[true][pred] += 1

        for true_class in range(len(class_names)):
            print(f"{class_names[true_class]:^10}", end="")
            for pred_class in range(len(class_names)):
                print(f"{confusion_matrix[true_class][pred_class]:^10}", end="")
            print()

        # Show a few example predictions
        print("\nExample predictions (5 samples):")
        num_examples = min(5, len(dataset))

        for i in range(num_examples):
            sample = dataset[i]
            features = sample["features"]
            true_label = sample["label"].item()

            pred_class, probabilities = predict(trained_model, features, device)

            print(f"Sample {i+1}:")
            print(f"True class: {class_names[true_label]}")
            print(f"Predicted: {class_names[pred_class]}")

            for j, prob in enumerate(probabilities[0]):
                print(f"  {class_names[j]}: {prob:.4f}")
            print()

    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
