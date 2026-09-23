"""
Synthetic Inference Tampering and Replay Scenario Generator
"""
import uuid
from pathlib import Path
from backend.app.config import settings
from backend.app.inference.provenance import create_provenance_record
from backend.app.inference.schema import PreprocessingConfig, InferenceConfig

def generate_inference_tamper_scenarios(model_sha256: str):
    """Generate clean, tampered, and replayed inference records for testing."""
    img_path = settings.INFERENCE_DIR / "tamper_test_base.jpg"
    from PIL import Image, ImageDraw
    im = Image.new("RGB", (64, 64), color=(50, 80, 120))
    d = ImageDraw.Draw(im)
    d.rectangle([10, 10, 54, 54], fill=(200, 100, 50))
    im.save(img_path)

    # 1. Authentic record
    clean_rec = create_provenance_record(
        input_image_path=img_path,
        model_sha256=model_sha256,
        output_data={"predicted_class_id": 1, "confidence": 0.94, "category": "pedestrian"},
        custom_nonce="nonce_auth_001",
        custom_sequence=101
    )

    # 2. Tampered record (output flipped from pedestrian to vehicle)
    tampered_rec = clean_rec.model_copy(deep=True)
    tampered_rec.raw_output["predicted_class_id"] = 0
    tampered_rec.raw_output["category"] = "vehicle"
    tampered_rec.raw_output["confidence"] = 0.999

    # 3. Replay record (same nonce and sequence as clean_rec)
    replayed_rec = clean_rec.model_copy(deep=True)
    replayed_rec.record_id = f"REC-RPLY-{uuid.uuid4().hex[:8].upper()}"

    return {
        "authentic_record": clean_rec.model_dump(),
        "tampered_record": tampered_rec.model_dump(),
        "replayed_record": replayed_rec.model_dump()
    }

if __name__ == "__main__":
    dummy_sha = "e" * 64
    scenarios = generate_inference_tamper_scenarios(dummy_sha)
    print("Inference tamper scenarios generated.")
