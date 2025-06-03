import os
import torch
from data.dataset import ImageDataset, get_transforms
from models.classifier import get_model
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse


def predict_labels(
    model_path,
    val_img_dir,
    arch='resnet18',
    batch_size=64,
    output_file='label_val.txt',
    device='cuda' if torch.cuda.is_available() else 'cpu',
    loss_type='bcewithlogits'
):
    """
    Loads a trained model and performs inference on a test set (val_img).

    Args:
        model_path (str): Path to the trained model checkpoint (.pth).
        val_img_dir (str): Directory containing test images (no labels).
        arch (str): Model architecture to load.
        batch_size (int): Batch size for inference.
        output_file (str): Output path to save predicted labels (one per line).
        device (str): Device for inference ('cuda' or 'cpu').
        loss_type (str): Indicates if sigmoid should be applied here ("bcewithlogits") or in model ("bce").

    Returns:
        None — predictions saved to file.
    """
    # Load model
    model = get_model(arch=arch, pretrained=False)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # Load test dataset
    test_dataset = ImageDataset(image_dir=val_img_dir, label_file=None, transform=get_transforms("val"))
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Predict
    predictions = []
    # Inference loop
    with torch.no_grad():
        for images in tqdm(test_loader, desc="Predicting"):
            images = images.to(device)
            outputs = model(images)
            # Apply sigmoid if using logits-based model
            if loss_type == 'bcewithlogits':
                outputs = torch.sigmoid(outputs)
            preds = (outputs > 0.5).int().cpu().numpy().flatten()
            predictions.extend(preds)

    # Write predictions to output file
    with open(output_file, 'w') as f:
        for pred in predictions:
            f.write(f"{pred}\n")

    print(f"✅ Saved predictions to {output_file}")

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--val_img_dir', type=str, required=True)
    parser.add_argument('--arch', type=str, default='resnet18')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--output_file', type=str, default='label_val.txt')
    parser.add_argument('--loss', type=str, default='bcewithlogits', choices=['bce', 'bcewithlogits'],
                        help='Loss type: "bce" (model includes sigmoid), or "bcewithlogits" (apply sigmoid here)')
    args = parser.parse_args()

    predict_labels(
        model_path=args.model_path,
        val_img_dir=args.val_img_dir,
        arch=args.arch,
        batch_size=args.batch_size,
        output_file=args.output_file,
        loss_type=args.loss
    )