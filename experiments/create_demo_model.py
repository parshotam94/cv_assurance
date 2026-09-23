"""
Creates local demonstration PyTorch and ONNX computer vision models without internet access.
"""
import torch
import torch.nn as nn
from pathlib import Path
from backend.app.config import settings

class VisionTrustClassifier(nn.Module):
    """Simple 3-class convolutional computer vision classifier."""
    def __init__(self, num_classes: int = 3):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2)
        
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2)
        
        self.fc1 = nn.Linear(32 * 16 * 16, 64)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.relu3(self.fc1(x))
        x = self.fc2(x)
        return x

def create_demo_models():
    """Build and save reference and candidate PyTorch & ONNX models."""
    torch.manual_seed(42)
    models_dir = settings.DATA_DIR / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    # 1. Reference model (PyTorch)
    ref_model = VisionTrustClassifier(num_classes=3)
    ref_model.eval()
    ref_pt_path = models_dir / "demo_reference_model.pt"
    torch.save(ref_model, ref_pt_path)

    # 2. Export Reference model to ONNX
    ref_onnx_path = models_dir / "demo_reference_model.onnx"
    dummy_input = torch.randn(1, 3, 64, 64)
    torch.onnx.export(
        ref_model,
        dummy_input,
        str(ref_onnx_path),
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        dynamo=False
    )

    # 3. Candidate model with perturbed/modified weights (Simulating Fine-Tuning or Trojan Alteration)
    cand_model = VisionTrustClassifier(num_classes=3)
    cand_state = ref_model.state_dict()
    # Mutate selected weight tensor
    for k in cand_state:
        if "fc2.weight" in k:
            cand_state[k] = cand_state[k] + torch.randn_like(cand_state[k]) * 0.5
    cand_model.load_state_dict(cand_state)
    cand_model.eval()
    
    cand_pt_path = models_dir / "demo_candidate_model.pt"
    torch.save(cand_model, cand_pt_path)

    # 4. Substituted model (different architecture / parameter count)
    class AlternateClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 8, kernel_size=3)
            self.fc = nn.Linear(8 * 62 * 62, 3)
        def forward(self, x):
            return self.fc(self.conv(x).view(x.size(0), -1))

    sub_model = AlternateClassifier()
    sub_model.eval()
    sub_onnx_path = models_dir / "demo_substituted_model.onnx"
    torch.onnx.export(
        sub_model,
        dummy_input,
        str(sub_onnx_path),
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        dynamo=False
    )

    return {
        "reference_pt": str(ref_pt_path.resolve()),
        "reference_onnx": str(ref_onnx_path.resolve()),
        "candidate_pt": str(cand_pt_path.resolve()),
        "substituted_onnx": str(sub_onnx_path.resolve())
    }

if __name__ == "__main__":
    paths = create_demo_models()
    print("Demo models created successfully:")
    for k, p in paths.items():
        print(f"  {k}: {p}")
