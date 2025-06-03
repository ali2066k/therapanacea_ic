import random
from torch.utils.data import Subset, DataLoader
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

class TransformedSubset:
    """
    Wraps a PyTorch Subset and applies a transform dynamically on access.

    Args:
        subset (torch.utils.data.Subset): A subset of a dataset.
        transform (callable): Transform to apply to each sample during retrieval.
    """
    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        x, y = self.subset[idx]
        if self.transform:
            x = self.transform(x)
        return x, y


def get_train_val_dataloaders(dataset, train_transform, val_transform, batch_size=64, val_ratio=0.2, seed=42):
    """
    Splits the dataset into stratified train and validation sets and returns DataLoaders for each.

    Args:
        dataset (Dataset): Full dataset, must have `labels` attribute.
        train_transform (callable): Transform for training data.
        val_transform (callable): Transform for validation data.
        batch_size (int): Batch size for both loaders.
        val_ratio (float): Fraction of data to use for validation.
        seed (int): Random seed for reproducibility.

    Returns:
        tuple: (train_loader, val_loader)
    """
    if dataset.labels is None:
        raise ValueError("Dataset must contain labels for stratified split.")

    indices = list(range(len(dataset)))
    labels = dataset.labels

    splitter = StratifiedShuffleSplit(n_splits=1, test_size=val_ratio, random_state=seed)
    train_idx, val_idx = next(splitter.split(indices, labels))

    train_dataset = TransformedSubset(Subset(dataset, train_idx), transform=train_transform)
    val_dataset = TransformedSubset(Subset(dataset, val_idx), transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, val_loader


# def get_train_val_dataloaders(dataset, train_transform, val_transform, batch_size=64, val_ratio=0.2, seed=42):
#     """
#     Splits the dataset into train and validation sets and returns DataLoaders for each.
#
#     Args:
#         dataset (Dataset): Full dataset.
#         train_transform (callable): Transform for training data.
#         val_transform (callable): Transform for validation data.
#         batch_size (int): Batch size for both loaders.
#         val_ratio (float): Fraction of data to use for validation.
#         seed (int): Random seed for reproducibility.
#
#     Returns:
#         tuple: (train_loader, val_loader)
#     """
#     indices = list(range(len(dataset)))
#     train_idx, val_idx = train_test_split(indices, test_size=val_ratio, random_state=seed, shuffle=True)
#
#     train_dataset = TransformedSubset(Subset(dataset, train_idx), transform=train_transform)
#     val_dataset = TransformedSubset(Subset(dataset, val_idx), transform=val_transform)
#
#     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
#     val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
#
#     return train_loader, val_loader


def plot_metrics(history, save_path=None):
    """
    Plots training and validation metrics over epochs: loss, accuracy, F1, and HTER.

    Args:
        history (dict): Training log dictionary with keys: train_loss, val_loss, train_acc, val_acc, train_f1, val_f1, val_hter.
        save_path (str): Optional path to save the plot image.
    """
    epochs = range(1, len(history["train_loss"]) + 1)
    plt.figure(figsize=(16, 10))

    plt.subplot(2, 2, 1)
    plt.plot(epochs, history["train_loss"], label='Train Loss')
    plt.plot(epochs, history["val_loss"], label='Val Loss')
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(2, 2, 2)
    plt.plot(epochs, history["train_acc"], label='Train Accuracy')
    plt.plot(epochs, history["val_acc"], label='Val Accuracy')
    plt.title("Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.subplot(2, 2, 3)
    plt.plot(epochs, history["train_f1"], label='Train F1')
    plt.plot(epochs, history["val_f1"], label='Val F1')
    plt.title("F1 Score")
    plt.xlabel("Epoch")
    plt.ylabel("F1")
    plt.legend()

    plt.subplot(2, 2, 4)
    plt.plot(epochs, history["val_hter"], label='Val HTER', color='red')
    plt.title("Validation HTER")
    plt.xlabel("Epoch")
    plt.ylabel("HTER")
    plt.legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"📊 Training metrics plot saved to: {save_path}")

    plt.show()


def plot_hter(history, save_path=None):
    """
    Plots the HTER (Half Total Error Rate) across epochs.

    Args:
        history (dict): Dictionary containing 'val_hter' list.
        save_path (str): Optional path to save the plot image.
    """
    plt.figure(figsize=(6, 4))
    plt.plot(range(1, len(history["val_hter"]) + 1), history["val_hter"], label="Validation HTER", color='red')
    plt.xlabel("Epoch")
    plt.ylabel("HTER")
    plt.title("Validation HTER over Epochs")
    plt.grid(True)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"📉 HTER curve saved to {save_path}")
    plt.show()


def save_history_to_csv(history, path='training_log.csv'):
    """
    Saves training metrics history to a CSV file.

    Args:
        history (dict): Training log dictionary with metrics as keys and list of epoch values.
        path (str): Output path to save the CSV file.
    """
    df = pd.DataFrame(history)
    df.to_csv(path, index_label='epoch')
    print(f"🧾 Training history saved to {path}")