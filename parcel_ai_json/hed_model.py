"""
HED (Holistically-Nested Edge Detection) Model for Fence Detection

Based on "Holistically-Nested Edge Detection" (Xie & Tu, 2015)
Optimized for fence line detection from satellite imagery.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class HED(nn.Module):
    """
    HED model with VGG16 backbone for edge detection.

    Outputs multi-scale edge predictions which are fused for final result.

    Supports 4-channel input (RGB + Regrid parcel mask) by adapting first conv layer.
    """

    def __init__(self, pretrained=True, input_channels=3):
        """
        Args:
            pretrained: Use pretrained VGG16 weights
            input_channels: Number of input channels
                (3 for RGB only, 4 for RGB + parcel)
        """
        super(HED, self).__init__()

        # Load pretrained VGG16
        if pretrained:
            vgg16 = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        else:
            vgg16 = models.vgg16(weights=None)

        # If using 4 channels, modify the first conv layer
        if input_channels == 4:
            # Get the first conv layer (Conv2d(3, 64, kernel_size=3, padding=1))
            first_conv = vgg16.features[0]

            # Create new conv layer with 4 input channels
            new_first_conv = nn.Conv2d(
                4,
                64,
                kernel_size=first_conv.kernel_size,
                stride=first_conv.stride,
                padding=first_conv.padding,
                bias=(first_conv.bias is not None),
            )

            # Copy pretrained weights for RGB channels
            with torch.no_grad():
                new_first_conv.weight[:, :3, :, :] = first_conv.weight
                # Initialize 4th channel weights (average of RGB channels)
                new_first_conv.weight[:, 3:4, :, :] = first_conv.weight.mean(
                    dim=1, keepdim=True
                )
                if first_conv.bias is not None:
                    new_first_conv.bias = first_conv.bias

            # Replace first conv layer
            vgg16.features[0] = new_first_conv

        # Split VGG16 into 5 blocks
        self.block1 = nn.Sequential(*list(vgg16.features[:4]))  # Conv1_1, Conv1_2
        self.block2 = nn.Sequential(*list(vgg16.features[4:9]))  # Conv2_1, Conv2_2
        self.block3 = nn.Sequential(
            *list(vgg16.features[9:16])
        )  # Conv3_1, Conv3_2, Conv3_3
        self.block4 = nn.Sequential(
            *list(vgg16.features[16:23])
        )  # Conv4_1, Conv4_2, Conv4_3
        self.block5 = nn.Sequential(
            *list(vgg16.features[23:30])
        )  # Conv5_1, Conv5_2, Conv5_3

        # Side outputs - predict edges at each scale
        self.side1 = nn.Conv2d(64, 1, kernel_size=1)
        self.side2 = nn.Conv2d(128, 1, kernel_size=1)
        self.side3 = nn.Conv2d(256, 1, kernel_size=1)
        self.side4 = nn.Conv2d(512, 1, kernel_size=1)
        self.side5 = nn.Conv2d(512, 1, kernel_size=1)

        # Fuse layer - combine all scales
        self.fuse = nn.Conv2d(5, 1, kernel_size=1)

    def forward(self, x):
        """
        Forward pass through HED.

        Returns:
            Dictionary with 'fused' output and individual 'side1-5' predictions
        """
        h, w = x.shape[2:]

        # Forward through blocks
        block1 = self.block1(x)
        block2 = self.block2(block1)
        block3 = self.block3(block2)
        block4 = self.block4(block3)
        block5 = self.block5(block4)

        # Side outputs at different scales
        side1 = self.side1(block1)
        side2 = F.interpolate(
            self.side2(block2), size=(h, w), mode="bilinear", align_corners=True
        )
        side3 = F.interpolate(
            self.side3(block3), size=(h, w), mode="bilinear", align_corners=True
        )
        side4 = F.interpolate(
            self.side4(block4), size=(h, w), mode="bilinear", align_corners=True
        )
        side5 = F.interpolate(
            self.side5(block5), size=(h, w), mode="bilinear", align_corners=True
        )

        # Fuse all side outputs
        fuse = self.fuse(torch.cat([side1, side2, side3, side4, side5], dim=1))

        return {
            "fused": torch.sigmoid(fuse),
            "side1": torch.sigmoid(side1),
            "side2": torch.sigmoid(side2),
            "side3": torch.sigmoid(side3),
            "side4": torch.sigmoid(side4),
            "side5": torch.sigmoid(side5),
        }


class WeightedBCELoss(nn.Module):
    """
    Weighted Binary Cross Entropy Loss for edge detection.
    Handles class imbalance by weighting positive (edge) pixels more.
    """

    def __init__(self, pos_weight=10.0):
        super().__init__()
        self.pos_weight = pos_weight

    def forward(self, pred, target):
        # Binary cross entropy with positive class weighting
        target = target.float()

        # BCE loss with higher weight for positive (fence) pixels
        loss = F.binary_cross_entropy(pred, target, reduction="none")

        # Apply weighting
        weights = torch.where(target > 0.5, self.pos_weight, 1.0)
        weighted_loss = loss * weights

        return weighted_loss.mean()


class HEDLoss(nn.Module):
    """
    Multi-scale loss for HED training.
    Supervises all side outputs and fused output.
    """

    def __init__(self, pos_weight=10.0):
        super().__init__()
        self.criterion = WeightedBCELoss(pos_weight)

    def forward(self, outputs, target):
        # Loss on fused output
        fuse_loss = self.criterion(outputs["fused"].squeeze(1), target.float())

        # Loss on each side output
        side_losses = []
        for i in range(1, 6):
            side_loss = self.criterion(outputs[f"side{i}"].squeeze(1), target.float())
            side_losses.append(side_loss)

        # Total loss: weighted combination
        total_loss = fuse_loss + sum(side_losses) * 0.5

        return total_loss, fuse_loss


def calculate_edge_metrics(pred, target, threshold=0.5):
    """
    Calculate metrics for edge detection.

    Args:
        pred: Predicted edge probabilities (0-1)
        target: Ground truth edges (0 or 1)
        threshold: Threshold for binarizing predictions

    Returns:
        Dictionary with precision, recall, F1
    """
    pred_binary = (pred > threshold).float()
    target = target.float()

    tp = (pred_binary * target).sum()
    fp = (pred_binary * (1 - target)).sum()
    fn = ((1 - pred_binary) * target).sum()

    precision = tp / (tp + fp + 1e-7)
    recall = tp / (tp + fn + 1e-7)
    f1 = 2 * precision * recall / (precision + recall + 1e-7)

    return {"precision": precision.item(), "recall": recall.item(), "f1": f1.item()}


if __name__ == "__main__":
    # Test model
    print("Testing HED model...")

    model = HED(pretrained=False)
    x = torch.randn(2, 3, 512, 512)

    outputs = model(x)

    print(f"Input shape: {x.shape}")
    print(f"Fused output shape: {outputs['fused'].shape}")
    print(f"Side outputs: {[outputs[f'side{i}'].shape for i in range(1, 6)]}")

    # Test loss
    target = torch.rand(2, 512, 512) > 0.95  # Sparse edges
    criterion = HEDLoss(pos_weight=10.0)

    total_loss, fuse_loss = criterion(outputs, target)
    print("\nLoss test:")
    print(f"Total loss: {total_loss.item():.4f}")
    print(f"Fuse loss: {fuse_loss.item():.4f}")

    # Test metrics
    metrics = calculate_edge_metrics(outputs["fused"].squeeze(1), target.float())
    print("\nMetrics test:")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1: {metrics['f1']:.4f}")

    print("\nHED model test passed!")
