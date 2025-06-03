import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class ImageDataset(Dataset):
    """
        Custom PyTorch Dataset for loading grayscale facial images with optional binary labels.

        Args:
            image_dir (str): Path to the directory containing images.
            label_file (str or None): Path to the label file (one label per line). If None, returns only images.
            transform (callable or None): Optional torchvision transform to apply to each image.

        Returns:
            If labels exist: a tuple (image_tensor, label)
            If no labels: just image_tensor
        """
    def __init__(self, image_dir, label_file=None, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.filenames = sorted(os.listdir(image_dir))

        if label_file:
            with open(label_file, 'r') as f:
                self.labels = [int(line.strip()) for line in f if line.strip()]
        else:
            self.labels = None

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        """
            Loads and returns one image (and its label if available) given an index.

            Args:
                idx (int): Index of the item

            Returns:
                image_tensor (Tensor): Transformed image
                label (int): Optional binary label (0 or 1)
        """
        filename = self.filenames[idx]
        img_path = os.path.join(self.image_dir, filename)
        image = Image.open(img_path).convert('L')

        if self.transform:
            image = self.transform(image)

        if self.labels:
            label = self.labels[idx]
            return image, label
        else:
            return image


def get_transforms(mode='train'):
    """
    Returns torchvision transforms tailored to the specified mode.

    Args:
        mode (str): One of ['train', 'val', 'test']

    Returns:
        transforms.Compose: A composed transform pipeline
    """
    base = [
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ]

    if mode == 'train':
        aug = [
            transforms.RandomApply([
                transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3)
            ], p=0.8),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
            transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))
        ]
        return transforms.Compose(aug + base)

    return transforms.Compose(base)