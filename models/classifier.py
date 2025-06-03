import torch.nn as nn
from torchvision import models


def get_model(arch='resnet18', pretrained=True, use_sigmoid=False):
    """
    Returns a modified torchvision model for binary classification on grayscale images.

    Args:
        arch (str): Architecture name, one of:
                    ['resnet18', 'resnet50', 'mobilenet_v2', 'efficientnet_b0', 'vgg16', 'densenet121']
        pretrained (bool): Whether to load pretrained ImageNet weights.
        use_sigmoid (bool): If True, appends a Sigmoid activation after the final linear layer
                            (use when using BCELoss). If False, output is raw logits (for BCEWithLogitsLoss).

    Returns:
        model (torch.nn.Module): A modified model ready for grayscale binary classification.
    """
    weights_map = {
        'resnet18': models.ResNet18_Weights.DEFAULT if pretrained else None,
        'resnet50': models.ResNet50_Weights.DEFAULT if pretrained else None,
        'mobilenet_v2': models.MobileNet_V2_Weights.DEFAULT if pretrained else None,
        'efficientnet_b0': models.EfficientNet_B0_Weights.DEFAULT if pretrained else None,
        'vgg16': models.VGG16_Weights.DEFAULT if pretrained else None,
        'densenet121': models.DenseNet121_Weights.DEFAULT if pretrained else None,
    }

    if arch == 'resnet18':
        model = models.resnet18(weights=weights_map[arch])
        model.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=2, padding=3, bias=False)
        num_ftrs = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(p=0.2),  # Dropout before final layer
            nn.Linear(num_ftrs, 1),
        ) if not use_sigmoid else nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(num_ftrs, 1),
            nn.Sigmoid()
        )

    elif arch == 'resnet50':
        model = models.resnet50(weights=weights_map[arch])
        model.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=2, padding=3, bias=False)
        num_ftrs = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(p=0.2),  # Dropout before final layer
            nn.Linear(num_ftrs, 1),
        ) if not use_sigmoid else nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(num_ftrs, 1),
            nn.Sigmoid()
        )

    elif arch == 'mobilenet_v2':
        model = models.mobilenet_v2(weights=weights_map[arch])
        model.features[0][0] = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)
        num_ftrs = model.classifier[1].in_features
        model.fc = nn.Sequential(
            nn.Dropout(p=0.2),  # Dropout before final layer
            nn.Linear(num_ftrs, 1),
        ) if not use_sigmoid else nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(num_ftrs, 1),
            nn.Sigmoid()
        )

    elif arch == 'efficientnet_b0':
        model = models.efficientnet_b0(weights=weights_map[arch])
        model.features[0][0] = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)
        num_ftrs = model.classifier[1].in_features
        model.fc = nn.Sequential(
            nn.Dropout(p=0.2),  # Dropout before final layer
            nn.Linear(num_ftrs, 1),
        ) if not use_sigmoid else nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(num_ftrs, 1),
            nn.Sigmoid()
        )

    elif arch == 'vgg16':
        model = models.vgg16(weights=weights_map[arch])
        model.features[0] = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
        num_ftrs = model.classifier[6].in_features
        model.fc = nn.Sequential(
            nn.Dropout(p=0.2),  # Dropout before final layer
            nn.Linear(num_ftrs, 1),
        ) if not use_sigmoid else nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(num_ftrs, 1),
            nn.Sigmoid()
        )

    elif arch == 'densenet121':
        model = models.densenet121(weights=weights_map[arch])
        model.features.conv0 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        num_ftrs = model.classifier.in_features
        model.fc = nn.Sequential(
            nn.Dropout(p=0.2),  # Dropout before final layer
            nn.Linear(num_ftrs, 1),
        ) if not use_sigmoid else nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(num_ftrs, 1),
            nn.Sigmoid()
        )

    else:
        raise ValueError(f"Unsupported architecture: {arch}")

    return model