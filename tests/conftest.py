"""
Pytest Fixtures and Global Configuration for VisionTrust Assurance
"""
import sys
import os
import json
import shutil
from pathlib import Path
import pytest
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.database.database import Base, get_db
from backend.app.database.repository import Repository
from backend.app.main import app
from backend.app.core.signatures import generate_or_load_keypair

@pytest.fixture(scope="session", autouse=True)
def ensure_keys():
    """Ensure offline Ed25519 signing keys exist."""
    generate_or_load_keypair()

@pytest.fixture(scope="function")
def db_session(tmp_path):
    """Create a temporary SQLite database session for each test."""
    db_file = tmp_path / "test_visiontrust.db"
    engine = create_engine(f"sqlite:///{db_file}", connect_args={"check_same_thread": False})
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    
    Base.metadata.create_all(bind=engine)
    session = TestingSessionLocal()
    
    try:
        yield session
    finally:
        session.close()
        engine.dispose()

@pytest.fixture(scope="function")
def test_repo(db_session):
    """Repository instance with test DB session."""
    return Repository(db_session)

@pytest.fixture(scope="function")
def client(db_session):
    """FastAPI TestClient with overridden get_db dependency."""
    def override_get_db():
        try:
            yield db_session
        finally:
            pass

    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()

@pytest.fixture(scope="function")
def test_image(tmp_path):
    """Create a clean 64x64 RGB test image."""
    img_path = tmp_path / "test_image.png"
    img = Image.fromarray(np.random.randint(50, 200, (64, 64, 3), dtype=np.uint8))
    img.save(img_path)
    return img_path

@pytest.fixture(scope="function")
def coco_dataset_dir(tmp_path):
    """Generate a minimal valid COCO dataset directory."""
    ds_dir = tmp_path / "coco_dataset"
    img_dir = ds_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    images = []
    annotations = []
    categories = [{"id": 0, "name": "pedestrian"}, {"id": 1, "name": "vehicle"}]

    for i in range(10):
        img_name = f"sample_{i:04d}.png"
        img_p = img_dir / img_name
        # Solid colored or patterned image
        arr = np.full((64, 64, 3), fill_value=20 + i * 20, dtype=np.uint8)
        Image.fromarray(arr).save(img_p)

        images.append({
            "id": i,
            "file_name": f"images/{img_name}",
            "width": 64,
            "height": 64,
            "source_id": "source_alpha" if i < 5 else "source_beta",
            "batch_id": "batch_1",
            "contributor": "alice" if i < 5 else "bob"
        })

        annotations.append({
            "id": i,
            "image_id": i,
            "category_id": i % 2,
            "bbox": [10, 10, 40, 40]
        })

    coco_json = {
        "images": images,
        "annotations": annotations,
        "categories": categories
    }

    with open(ds_dir / "annotations.json", "w", encoding="utf-8") as f:
        json.dump(coco_json, f)

    return ds_dir

class TinyTestNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 4, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(4, 2)

    def forward(self, x):
        return self.fc(self.pool(torch.relu(self.conv(x))).flatten(1))

@pytest.fixture(scope="function")
def model_paths(tmp_path):
    """Generate minimal PyTorch and ONNX models for testing."""
    model = TinyTestNet()
    model.eval()

    pt_path = tmp_path / "tiny_model.pt"
    torch.save(model.state_dict(), pt_path)

    onnx_path = tmp_path / "tiny_model.onnx"
    dummy_input = torch.randn(1, 3, 64, 64)
    torch.onnx.export(
        model,
        dummy_input,
        str(onnx_path),
        input_names=["input"],
        output_names=["output"],
        dynamo=False
    )

    return {
        "pytorch": pt_path,
        "onnx": onnx_path
    }
