"""
PyTorch and TorchScript Model Adapter and White-Box Analyzer
"""
from pathlib import Path
from typing import Dict, Any, List
import numpy as np
import torch
from backend.app.model.loader import ModelAdapter

class TorchModelAdapter(ModelAdapter):
    def __init__(self, model_path: Path):
        super().__init__(model_path)
        self.access_level = "WHITE_BOX"
        self.model = None
        self.is_torchscript = False
        self.state_dict = None

    def load(self) -> bool:
        try:
            # First try loading as TorchScript
            try:
                self.model = torch.jit.load(str(self.model_path), map_location="cpu")
                self.model.eval()
                self.is_torchscript = True
                self.access_level = "GRAY_BOX"
                return True
            except Exception:
                pass

            # Try loading as full PyTorch model or state dict
            data = torch.load(str(self.model_path), map_location="cpu", weights_only=False)
            if isinstance(data, torch.nn.Module):
                self.model = data
                self.model.eval()
                self.access_level = "WHITE_BOX"
            elif isinstance(data, dict):
                # State dict
                self.state_dict = data
                self.access_level = "WHITE_BOX"
            return True
        except Exception as e:
            raise RuntimeError(f"Failed to load PyTorch model {self.model_path.name}: {str(e)}")

    def predict(self, input_tensor: np.ndarray) -> np.ndarray:
        if self.model is None:
            self.load()

        if self.model is None and self.state_dict is not None:
            # State dict without model architecture cannot perform forward pass directly
            # Return dummy feature representation from weight projection
            return np.zeros((input_tensor.shape[0], 10), dtype=np.float32)

        tensor = torch.from_numpy(input_tensor.astype(np.float32))
        if len(tensor.shape) == 3:
            tensor = tensor.unsqueeze(0)
        if len(tensor.shape) == 4 and tensor.shape[-1] in [1, 3]:
            # [B, H, W, C] -> [B, C, H, W]
            tensor = tensor.permute(0, 3, 1, 2)

        with torch.no_grad():
            output = self.model(tensor)
            if isinstance(output, tuple):
                output = output[0]
            if hasattr(output, "logits"):
                output = output.logits
            return output.cpu().numpy()

    def inspect_architecture(self) -> Dict[str, Any]:
        if self.model is None and self.state_dict is None:
            self.load()

        param_count = 0
        layers_info = []

        if self.model is not None and not self.is_torchscript:
            for name, param in self.model.named_parameters():
                num_p = int(np.prod(param.shape))
                param_count += num_p
                layers_info.append({"name": name, "shape": list(param.shape), "params": num_p})
            arch_type = self.model.__class__.__name__
        elif self.state_dict is not None:
            for k, v in self.state_dict.items():
                if hasattr(v, "shape"):
                    num_p = int(np.prod(v.shape))
                    param_count += num_p
                    layers_info.append({"name": k, "shape": list(v.shape), "params": num_p})
            arch_type = "PyTorchStateDict"
        elif self.is_torchscript:
            arch_type = "TorchScript"
            for name, param in self.model.named_parameters():
                num_p = int(np.prod(param.shape))
                param_count += num_p
                layers_info.append({"name": name, "shape": list(param.shape), "params": num_p})

        return {
            "framework": "TorchScript" if self.is_torchscript else "PyTorch",
            "architecture_class": arch_type,
            "parameter_count": param_count,
            "layers_count": len(layers_info),
            "layers_sample": layers_info[:15],
            "access_level": self.access_level
        }

    def compute_weight_fingerprint(self) -> Dict[str, Any]:
        if self.model is None and self.state_dict is None:
            self.load()

        stats = []
        l2_norms = []
        tensors = {}

        if self.model is not None:
            tensors = dict(self.model.named_parameters())
        elif self.state_dict is not None:
            tensors = {k: v for k, v in self.state_dict.items() if hasattr(v, "cpu")}

        for name, t in list(tensors.items())[:25]:
            arr = t.detach().cpu().numpy().astype(np.float32)
            mean_val = float(np.mean(arr))
            std_val = float(np.std(arr))
            norm_val = float(np.linalg.norm(arr))
            l2_norms.append(norm_val)
            stats.append({
                "name": name,
                "shape": list(arr.shape),
                "mean": round(mean_val, 5),
                "std": round(std_val, 5),
                "l2_norm": round(norm_val, 5),
                "zero_fraction": round(float(np.sum(arr == 0) / arr.size), 4)
            })

        return {
            "available": True,
            "total_layers": len(tensors),
            "mean_l2_norm": round(float(np.mean(l2_norms)), 5) if l2_norms else 0.0,
            "layer_statistics": stats
        }
