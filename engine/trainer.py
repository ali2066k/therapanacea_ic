import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from tqdm import tqdm
import os


def compute_hter(y_true, y_pred):
    """
    Computes the Half Total Error Rate (HTER) from binary predictions.

    HTER = 0.5 * (False Rejection Rate + False Acceptance Rate)

    Args:
        y_true (list or array): Ground truth binary labels (0 or 1).
        y_pred (list or array): Predicted binary labels.

    Returns:
        float: HTER score. Returns 1.0 if confusion matrix shape is invalid.
    """
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape != (2, 2):
        return 1.0
    fnr = cm[1, 0] / (cm[1, 0] + cm[1, 1])
    fpr = cm[0, 1] / (cm[0, 0] + cm[0, 1])
    return 0.5 * (fnr + fpr)


def train_model(
    model, train_loader, val_loader, device,
    num_epochs=10, lr=1e-4,
    save_best_model_path=None,
    early_stopping_patience=5,
    loss_type='bcewithlogits'
):
    """
    Trains a binary classification model and tracks metrics including HTER.

    Args:
        model (torch.nn.Module): The model to train.
        train_loader (DataLoader): DataLoader for training data.
        val_loader (DataLoader): DataLoader for validation data.
        device (torch.device): CUDA or CPU.
        num_epochs (int): Number of epochs to train.
        lr (float): Learning rate.
        save_best_model_path (str or None): Where to save the best model (lowest HTER).
        early_stopping_patience (int): Stop training if no HTER improvement for N epochs.
        loss_type (str): One of 'bce' (with Sigmoid in model) or 'bcewithlogits' (preferred, no Sigmoid).

    Returns:
        dict: Training history dictionary with metrics per epoch.
    """
    # Number of samples for each class (from label distribution analysis)
    neg_count = 12102  # Count of class 0 (negative samples)
    pos_count = 87898  # Count of class 1 (positive samples)
    imbalance_ratio = neg_count / pos_count  # Gives higher weight to the minority class

    # Create appropriate loss function
    if loss_type == 'bce':
        criterion = nn.BCELoss()
    else:
        pos_weight = torch.tensor([imbalance_ratio]).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)


    best_hter = float('inf')
    patience_counter = 0

    history = {
        "train_loss": [], "train_acc": [], "train_f1": [],
        "val_loss": [], "val_acc": [], "val_f1": [], "val_hter": []
    }

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        all_train_preds, all_train_labels = [], []

        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]", leave=False)
        for images, labels in loop:
            images = images.to(device)
            labels = labels.float().unsqueeze(1).to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            preds = (outputs > 0.5).float()
            all_train_preds.extend(preds.cpu().numpy())
            all_train_labels.extend(labels.cpu().numpy())

            loop.set_postfix(loss=loss.item())

        # Metrics
        epoch_train_loss = train_loss / len(train_loader.dataset)
        epoch_train_acc = accuracy_score(all_train_labels, all_train_preds)
        epoch_train_f1 = f1_score(all_train_labels, all_train_preds)
        history["train_loss"].append(epoch_train_loss)
        history["train_acc"].append(epoch_train_acc)
        history["train_f1"].append(epoch_train_f1)

        # Validation
        model.eval()
        val_loss = 0.0
        all_val_preds, all_val_labels = [], []

        loop = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]", leave=False)
        with torch.no_grad():
            for images, labels in loop:
                images = images.to(device)
                labels = labels.float().unsqueeze(1).to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)

                preds = (outputs > 0.5).float()
                all_val_preds.extend(preds.cpu().numpy())
                all_val_labels.extend(labels.cpu().numpy())
                loop.set_postfix(loss=loss.item())

        epoch_val_loss = val_loss / len(val_loader.dataset)
        epoch_val_acc = accuracy_score(all_val_labels, all_val_preds)
        epoch_val_f1 = f1_score(all_val_labels, all_val_preds)
        epoch_val_hter = compute_hter(all_val_labels, all_val_preds)

        # Step scheduler after validation loss is computed
        scheduler.step(epoch_val_loss)

        history["val_loss"].append(epoch_val_loss)
        history["val_acc"].append(epoch_val_acc)
        history["val_f1"].append(epoch_val_f1)
        history["val_hter"].append(epoch_val_hter)

        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print(f"  🟢 Train — Loss: {epoch_train_loss:.4f}, Acc: {epoch_train_acc:.4f}, F1: {epoch_train_f1:.4f}")
        print(f"  🔵 Val   — Loss: {epoch_val_loss:.4f}, Acc: {epoch_val_acc:.4f}, F1: {epoch_val_f1:.4f}, HTER: {epoch_val_hter:.4f}")

        # ✅ Save best model
        if epoch_val_hter < best_hter:
            best_hter = epoch_val_hter
            patience_counter = 0
            if save_best_model_path:
                os.makedirs(os.path.dirname(save_best_model_path), exist_ok=True)
                torch.save(model.state_dict(), save_best_model_path)
                print(f"📌 Best model saved (HTER={best_hter:.4f}) to {save_best_model_path}")
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                print(f"⏹️ Early stopping triggered at epoch {epoch+1} (no improvement for {early_stopping_patience} epochs)")
                break

    return history