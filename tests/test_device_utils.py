#!/usr/bin/env python3
"""Tests for device detection utilities."""

import unittest
from unittest.mock import patch
import sys

from parcel_ai_json.device_utils import get_best_device


class TestDeviceUtils(unittest.TestCase):
    """Test device detection utilities."""

    @patch("torch.cuda.is_available")
    def test_get_best_device_cuda(self, mock_cuda_available):
        """Test CUDA device detection."""
        mock_cuda_available.return_value = True

        device = get_best_device()

        self.assertEqual(device, "cuda")
        mock_cuda_available.assert_called_once()

    @patch("torch.backends.mps.is_available")
    @patch("torch.cuda.is_available")
    def test_get_best_device_mps(self, mock_cuda_available, mock_mps_available):
        """Test MPS (Apple Silicon) device detection."""
        # CUDA not available
        mock_cuda_available.return_value = False

        # MPS available
        mock_mps_available.return_value = True

        device = get_best_device()

        self.assertEqual(device, "mps")

    @patch("torch.backends.mps.is_available")
    @patch("torch.cuda.is_available")
    def test_get_best_device_cpu_fallback(
        self, mock_cuda_available, mock_mps_available
    ):
        """Test CPU fallback when no GPU available."""
        # Neither CUDA nor MPS available
        mock_cuda_available.return_value = False
        mock_mps_available.return_value = False

        device = get_best_device()

        self.assertEqual(device, "cpu")

    @patch("torch.cuda.is_available")
    def test_get_best_device_no_mps_backend(self, mock_cuda_available):
        """Test CPU fallback when MPS backend doesn't exist."""
        # CUDA not available
        mock_cuda_available.return_value = False

        # Remove MPS backend to simulate older PyTorch
        import torch

        original_mps = getattr(torch.backends, "mps", None)
        if hasattr(torch.backends, "mps"):
            delattr(torch.backends, "mps")

        try:
            device = get_best_device()
            self.assertEqual(device, "cpu")
        finally:
            # Restore MPS backend
            if original_mps is not None:
                torch.backends.mps = original_mps

    def test_get_best_device_no_torch(self):
        """Test CPU fallback when torch not installed."""
        # Temporarily hide torch module
        torch_module = sys.modules.get("torch")
        if "torch" in sys.modules:
            del sys.modules["torch"]

        try:
            with patch.dict("sys.modules", {"torch": None}):
                device = get_best_device()
                self.assertEqual(device, "cpu")
        finally:
            # Restore torch module
            if torch_module is not None:
                sys.modules["torch"] = torch_module


if __name__ == "__main__":
    unittest.main()
