"""Tests for HED (Holistically-Nested Edge Detection) model."""

import unittest
import torch
from unittest.mock import patch, Mock

from parcel_ai_json.hed_model import (
    HED,
    WeightedBCELoss,
    HEDLoss,
    calculate_edge_metrics,
)


class TestHEDModel(unittest.TestCase):
    """Test HED model architecture and components."""

    def test_hed_init_3_channels(self):
        """Test HED model initialization with 3 input channels (RGB)."""
        model = HED(pretrained=False, input_channels=3)

        self.assertIsNotNone(model.block1)
        self.assertIsNotNone(model.block2)
        self.assertIsNotNone(model.block3)
        self.assertIsNotNone(model.block4)
        self.assertIsNotNone(model.block5)
        self.assertIsNotNone(model.side1)
        self.assertIsNotNone(model.side2)
        self.assertIsNotNone(model.side3)
        self.assertIsNotNone(model.side4)
        self.assertIsNotNone(model.side5)
        self.assertIsNotNone(model.fuse)

    def test_hed_init_4_channels(self):
        """Test HED model initialization with 4 input channels (RGB + mask)."""
        model = HED(pretrained=False, input_channels=4)

        # Check that first conv layer has 4 input channels
        first_conv = model.block1[0]
        self.assertEqual(first_conv.in_channels, 4)
        self.assertEqual(first_conv.out_channels, 64)

    @patch("parcel_ai_json.hed_model.models.vgg16")
    def test_hed_init_pretrained(self, mock_vgg16):
        """Test HED model initialization with pretrained weights."""
        mock_model = Mock()
        mock_model.features = torch.nn.Sequential(
            torch.nn.Conv2d(3, 64, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(64, 64, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            # Block 2
            torch.nn.Conv2d(64, 128, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(128, 128, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            # Block 3
            torch.nn.Conv2d(128, 256, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(256, 256, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(256, 256, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            # Block 4
            torch.nn.Conv2d(256, 512, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(512, 512, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(512, 512, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            # Block 5
            torch.nn.Conv2d(512, 512, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(512, 512, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(512, 512, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
        )
        mock_vgg16.return_value = mock_model

        model = HED(pretrained=True, input_channels=3)
        mock_vgg16.assert_called_once()

    def test_hed_forward_3_channels(self):
        """Test HED forward pass with 3 channels."""
        model = HED(pretrained=False, input_channels=3)
        model.eval()

        # Create test input (batch=2, channels=3, height=128, width=128)
        x = torch.randn(2, 3, 128, 128)

        with torch.no_grad():
            outputs = model(x)

        # Check output structure
        self.assertIn("fused", outputs)
        self.assertIn("side1", outputs)
        self.assertIn("side2", outputs)
        self.assertIn("side3", outputs)
        self.assertIn("side4", outputs)
        self.assertIn("side5", outputs)

        # Check output shapes (all should match input spatial dimensions)
        self.assertEqual(outputs["fused"].shape, (2, 1, 128, 128))
        self.assertEqual(outputs["side1"].shape, (2, 1, 128, 128))
        self.assertEqual(outputs["side2"].shape, (2, 1, 128, 128))
        self.assertEqual(outputs["side3"].shape, (2, 1, 128, 128))
        self.assertEqual(outputs["side4"].shape, (2, 1, 128, 128))
        self.assertEqual(outputs["side5"].shape, (2, 1, 128, 128))

        # Check outputs are in [0, 1] range (sigmoid applied)
        self.assertTrue(torch.all(outputs["fused"] >= 0))
        self.assertTrue(torch.all(outputs["fused"] <= 1))

    def test_hed_forward_4_channels(self):
        """Test HED forward pass with 4 channels."""
        model = HED(pretrained=False, input_channels=4)
        model.eval()

        # Create test input (batch=1, channels=4, height=64, width=64)
        x = torch.randn(1, 4, 64, 64)

        with torch.no_grad():
            outputs = model(x)

        # Check output structure and shapes
        self.assertIn("fused", outputs)
        self.assertEqual(outputs["fused"].shape, (1, 1, 64, 64))


class TestWeightedBCELoss(unittest.TestCase):
    """Test Weighted BCE Loss for edge detection."""

    def test_weighted_bce_loss_init(self):
        """Test WeightedBCELoss initialization."""
        loss_fn = WeightedBCELoss(pos_weight=5.0)
        self.assertEqual(loss_fn.pos_weight, 5.0)

    def test_weighted_bce_loss_forward(self):
        """Test WeightedBCELoss forward pass."""
        loss_fn = WeightedBCELoss(pos_weight=10.0)

        # Create predictions and targets
        pred = torch.sigmoid(torch.randn(2, 64, 64))
        target = (torch.rand(2, 64, 64) > 0.9).float()

        loss = loss_fn(pred, target)

        # Check loss is a scalar tensor
        self.assertEqual(loss.shape, torch.Size([]))
        self.assertTrue(loss.item() >= 0)

    def test_weighted_bce_loss_balancing(self):
        """Test that positive class gets higher weight."""
        loss_fn = WeightedBCELoss(pos_weight=10.0)

        # Perfect prediction on negative class
        pred_neg = torch.zeros(1, 10, 10)
        target_neg = torch.zeros(1, 10, 10)
        loss_neg = loss_fn(pred_neg, target_neg)

        # Prediction error on positive class
        pred_pos = torch.zeros(1, 10, 10)
        target_pos = torch.ones(1, 10, 10)
        loss_pos = loss_fn(pred_pos, target_pos)

        # Positive class error should have higher loss
        self.assertTrue(loss_pos.item() > loss_neg.item())


class TestHEDLoss(unittest.TestCase):
    """Test multi-scale HED loss."""

    def test_hed_loss_init(self):
        """Test HEDLoss initialization."""
        loss_fn = HEDLoss(pos_weight=10.0)
        self.assertIsInstance(loss_fn.criterion, WeightedBCELoss)
        self.assertEqual(loss_fn.criterion.pos_weight, 10.0)

    def test_hed_loss_forward(self):
        """Test HEDLoss forward pass."""
        loss_fn = HEDLoss(pos_weight=10.0)

        # Create mock outputs from HED model
        outputs = {
            "fused": torch.sigmoid(torch.randn(2, 1, 64, 64)),
            "side1": torch.sigmoid(torch.randn(2, 1, 64, 64)),
            "side2": torch.sigmoid(torch.randn(2, 1, 64, 64)),
            "side3": torch.sigmoid(torch.randn(2, 1, 64, 64)),
            "side4": torch.sigmoid(torch.randn(2, 1, 64, 64)),
            "side5": torch.sigmoid(torch.randn(2, 1, 64, 64)),
        }

        target = (torch.rand(2, 64, 64) > 0.95).float()

        total_loss, fuse_loss = loss_fn(outputs, target)

        # Check losses are scalar tensors
        self.assertEqual(total_loss.shape, torch.Size([]))
        self.assertEqual(fuse_loss.shape, torch.Size([]))
        self.assertTrue(total_loss.item() >= 0)
        self.assertTrue(fuse_loss.item() >= 0)

        # Total loss should be >= fuse loss (includes side outputs)
        self.assertTrue(total_loss.item() >= fuse_loss.item())


class TestEdgeMetrics(unittest.TestCase):
    """Test edge detection metrics calculation."""

    def test_calculate_edge_metrics_perfect(self):
        """Test metrics with perfect prediction."""
        pred = torch.ones(10, 10)
        target = torch.ones(10, 10)

        metrics = calculate_edge_metrics(pred, target, threshold=0.5)

        self.assertAlmostEqual(metrics["precision"], 1.0, places=5)
        self.assertAlmostEqual(metrics["recall"], 1.0, places=5)
        self.assertAlmostEqual(metrics["f1"], 1.0, places=5)

    def test_calculate_edge_metrics_zero(self):
        """Test metrics with all zeros."""
        pred = torch.zeros(10, 10)
        target = torch.zeros(10, 10)

        metrics = calculate_edge_metrics(pred, target, threshold=0.5)

        # With no positives, precision/recall should be 0
        self.assertTrue(0 <= metrics["precision"] <= 1)
        self.assertTrue(0 <= metrics["recall"] <= 1)
        self.assertTrue(0 <= metrics["f1"] <= 1)

    def test_calculate_edge_metrics_partial(self):
        """Test metrics with partial match."""
        # Half of predictions are correct
        pred = torch.zeros(10, 10)
        pred[:5, :] = 1.0  # Top half predicted as edge

        target = torch.zeros(10, 10)
        target[:, :5] = 1.0  # Left half is edge

        metrics = calculate_edge_metrics(pred, target, threshold=0.5)

        # Should have some precision and recall
        self.assertTrue(0 < metrics["precision"] < 1)
        self.assertTrue(0 < metrics["recall"] < 1)
        self.assertTrue(0 < metrics["f1"] < 1)

    def test_calculate_edge_metrics_threshold(self):
        """Test that threshold affects binarization."""
        pred = torch.full((10, 10), 0.6)
        target = torch.ones(10, 10)

        # With threshold=0.5, all predictions should be positive
        metrics_low = calculate_edge_metrics(pred, target, threshold=0.5)
        self.assertAlmostEqual(metrics_low["precision"], 1.0, places=5)

        # With threshold=0.7, all predictions should be negative
        metrics_high = calculate_edge_metrics(pred, target, threshold=0.7)
        self.assertAlmostEqual(metrics_high["recall"], 0.0, places=5)


class TestHEDIntegration(unittest.TestCase):
    """Integration tests for HED model end-to-end."""

    def test_hed_training_step_simulation(self):
        """Simulate a training step with HED model."""
        model = HED(pretrained=False, input_channels=3)
        criterion = HEDLoss(pos_weight=10.0)

        # Create batch
        x = torch.randn(2, 3, 64, 64)
        target = (torch.rand(2, 64, 64) > 0.95).float()

        # Forward pass
        outputs = model(x)
        total_loss, fuse_loss = criterion(outputs, target)

        # Check loss can be backpropagated
        self.assertTrue(total_loss.requires_grad)

        # Simulate backward pass (don't actually update weights)
        total_loss.backward()

        # Check gradients exist
        for param in model.parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad)

    def test_hed_inference_mode(self):
        """Test HED model in inference mode."""
        model = HED(pretrained=False, input_channels=3)
        model.eval()

        x = torch.randn(1, 3, 128, 128)

        with torch.no_grad():
            outputs = model(x)

        # Check outputs don't require gradients
        self.assertFalse(outputs["fused"].requires_grad)

        # Check all outputs are valid probabilities
        for key in ["fused", "side1", "side2", "side3", "side4", "side5"]:
            self.assertTrue(torch.all(outputs[key] >= 0))
            self.assertTrue(torch.all(outputs[key] <= 1))


if __name__ == "__main__":
    unittest.main()
