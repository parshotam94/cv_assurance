"""
Model adapter base class and model loader factory
Supports ONNX, PyTorch (.pt, .pth), and TorchScript (.ts)
"""
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np

class ModelAdapter(ABC):
    """Abstract model adapter interface for model-agnostic assurance."""

    def __init__(self, model_path: Path):
        self.model_path = model_path
        self.access_level = "BLACK_BOX"

    @abstractmethod
    def load(self) -> bool:
        """Load model into memory or initialize runtime session."""
        pass

    @abstractmethod
    def predict(self, input_tensor: np.ndarray) -> np.ndarray:
        """Run inference on input tensor (shape: [B, C, H, W] or [B, H, W, C]). Returns output array."""
        pass

    @abstractmethod
    def inspect_architecture(self) -> Dict[str, Any]:
        """Inspect architecture, layers, inputs, outputs, and parameter count."""
        pass

    @abstractmethod
    def compute_weight_fingerprint(self) -> Dict[str, Any]:
        """Compute tensor statistics (mean, std, norms) if white-box access is available."""
        pass

def get_model_adapter(model_path: Path) -> ModelAdapter:
    """Factory to instantiate appropriate model adapter based on file extension and format."""
    ext = model_path.suffix.lower()
    
    if ext == ".onnx":
        from backend.app.model.onnx_analyzer import ONNXModelAdapter
        return ONNXModelAdapter(model_path)
    elif ext in [".pt", ".pth", ".ts"]:
        from backend.app.model.torch_analyzer import TorchModelAdapter
        return TorchModelAdapter(model_path)
    else:
        raise ValueError(f"Unsupported model format: '{ext}'. Supported: .onnx, .pt, .pth, .ts")
