import argparse
import os
import torch
from data.dataset import ImageDataset, get_transforms
from data.utils import get_train_val_dataloaders, plot_metrics
from models.classifier import get_model
from engine.trainer import train_model
from data.utils import plot_hter, save_history_to_csv


def parse_args():
    """
    Parses command-line arguments for data, model, training config, and output options.
    """
    parser = argparse.ArgumentParser(description="TheraPanacea Image Classification Pipeline")

    # Data paths
    parser.add_argument('--image_dir', type=str, required=True, help='Path to training images')
    parser.add_argument('--label_file', type=str, required=True, help='Path to label_train.txt')

    # Model and training
    parser.add_argument('--arch', type=str, default='resnet18', help='Model architecture')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')

    # Model IO
    parser.add_argument('--save_model', type=str, default=None, help='Path to save the trained model')
    parser.add_argument('--load_model', type=str, default=None, help='Path to a model checkpoint to load before training')

    # Plot IO
    parser.add_argument('--save_plot', type=str, default=None, help='Path to save the training metrics plot')
    parser.add_argument('--gpu', type=int, default=0, help='GPU index to use (e.g., 0, 1, ...)')

    parser.add_argument('--save_best_model', type=str, default=None, help='Path to save the best model (lowest HTER)')
    parser.add_argument('--save_hter_plot', type=str, default=None, help='Path to save the HTER curve plot')
    parser.add_argument('--early_stop', type=int, default=5, help='Early stopping patience')

    parser.add_argument(
        '--loss', type=str, default='bcewithlogits',
        choices=['bce', 'bcewithlogits'],
        help='Loss function: bce (use Sigmoid + BCELoss) or bcewithlogits (preferred, no Sigmoid)'
    )

    return parser.parse_args()

def main():
    args = parse_args()
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Using device: {device}")
    if device.type == 'cuda':
        print(f"🧠 GPU name: {torch.cuda.get_device_name(args.gpu)}")

    # Create output directories if needed
    if args.save_model:
        os.makedirs(os.path.dirname(args.save_model), exist_ok=True)
    if args.save_plot:
        os.makedirs(os.path.dirname(args.save_plot), exist_ok=True)

    # Load dataset and create train/val loaders
    base_dataset = ImageDataset(image_dir=args.image_dir, label_file=args.label_file)

    train_loader, val_loader = get_train_val_dataloaders(
        dataset=base_dataset,
        train_transform=get_transforms("train"),
        val_transform=get_transforms("val"),
        batch_size=args.batch_size
    )

    # Initialize model with sigmoid if using BCE loss
    use_sigmoid = args.loss == 'bce'
    model = get_model(arch=args.arch, pretrained=True, use_sigmoid=use_sigmoid).to(device)

    if args.load_model:
        model.load_state_dict(torch.load(args.load_model, map_location=device))
        print(f"✅ Loaded model from {args.load_model}")

    # Train the model
    history = train_model(
        model,
        train_loader,
        val_loader,
        device=device,
        num_epochs=args.epochs,
        lr=args.lr,
        save_best_model_path=args.save_best_model,
        early_stopping_patience=args.early_stop
    )

    # Save plots and logs
    plot_metrics(history, save_path=args.save_plot)
    plot_hter(history, save_path=args.save_hter_plot)

    # Save final model
    if args.save_model:
        torch.save(model.state_dict(), args.save_model)
        print(f"💾 Model saved to {args.save_model}")

    # Save full training log
    # Create logs directory if needed
    os.makedirs('./logs', exist_ok=True)
    save_history_to_csv(history, path='./logs/training_log.csv')


if __name__ == "__main__":
    main()
