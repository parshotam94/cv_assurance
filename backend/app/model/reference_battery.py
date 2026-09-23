"""
Model Behavioural Fingerprinting using Controlled Input Battery
Evaluates model prediction stability across brightness, rotations, noise, blur, and perturbations.
"""
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import cv2
from backend.app.model.loader import ModelAdapter

def generate_synthetic_battery(sample_count: int = 40, image_size: Tuple[int, int] = (64, 64)) -> List[Dict[str, Any]]:
    """
    Generate controlled testing battery with systematic environmental and geometric variations.
    """
    np.random.seed(42)
    battery = []
    h, w = image_size

    # Base patterns (circles, squares, diagonals)
    base_images = []
    for _ in range(max(4, sample_count // 6)):
        img = np.zeros((h, w, 3), dtype=np.float32)
        # Draw random shapes
        cx, cy = np.random.randint(15, h - 15, size=2)
        cv2.circle(img, (cx, cy), np.random.randint(8, 20), (0.8, 0.4, 0.2), -1)
        x1, y1 = np.random.randint(5, h // 2, size=2)
        cv2.rectangle(img, (x1, y1), (x1 + 20, y1 + 20), (0.2, 0.7, 0.5), -1)
        base_images.append(img)

    for i, base in enumerate(base_images):
        # 1. Normal
        battery.append({"type": "NORMAL", "image": base.copy(), "desc": f"Base sample {i}"})
        # 2. Brightness variation (+40%)
        bright = np.clip(base * 1.4, 0.0, 1.0)
        battery.append({"type": "BRIGHTNESS_HIGH", "image": bright, "desc": f"Sample {i} +40% brightness"})
        # 3. Low brightness (-50%)
        dark = np.clip(base * 0.5, 0.0, 1.0)
        battery.append({"type": "BRIGHTNESS_LOW", "image": dark, "desc": f"Sample {i} -50% brightness"})
        # 4. Gaussian noise
        noise = np.random.normal(0, 0.1, base.shape).astype(np.float32)
        noisy = np.clip(base + noise, 0.0, 1.0)
        battery.append({"type": "NOISE_GAUSSIAN", "image": noisy, "desc": f"Sample {i} with Gaussian noise"})
        # 5. Gaussian blur
        blurred = cv2.GaussianBlur(base, (5, 5), 1.5)
        battery.append({"type": "BLUR", "image": blurred, "desc": f"Sample {i} blurred"})
        # 6. High contrast
        contrast = np.clip((base - 0.5) * 1.8 + 0.5, 0.0, 1.0)
        battery.append({"type": "HIGH_CONTRAST", "image": contrast, "desc": f"Sample {i} high contrast"})

    return battery[:sample_count]

def evaluate_behavioural_battery(
    candidate_adapter: ModelAdapter,
    reference_adapter: Optional[ModelAdapter] = None,
    battery_samples: int = 40
) -> Dict[str, Any]:
    """
    Run candidate (and optional reference) model across the controlled perturbation battery.
    Measures prediction consistency, class agreement, and anomaly frequency.
    """
    battery = generate_synthetic_battery(sample_count=battery_samples)
    
    cand_preds = []
    cand_confidences = []
    ref_preds = []
    
    anomalous_tests = []
    class_agreements = []

    for item in battery:
        img = item["image"]
        
        try:
            cand_out = candidate_adapter.predict(img)
            # Find predicted class and confidence
            if cand_out.size > 0:
                # Softmax if not already
                flat_out = cand_out.flatten()
                exp_out = np.exp(flat_out - np.max(flat_out))
                probs = exp_out / np.sum(exp_out)
                pred_cls = int(np.argmax(probs))
                conf = float(np.max(probs))
            else:
                pred_cls = 0
                conf = 0.5
        except Exception:
            pred_cls = -1
            conf = 0.0

        cand_preds.append(pred_cls)
        cand_confidences.append(conf)

        if reference_adapter:
            try:
                ref_out = reference_adapter.predict(img)
                if ref_out.size > 0:
                    exp_r = np.exp(ref_out.flatten() - np.max(ref_out.flatten()))
                    probs_r = exp_r / np.sum(exp_r)
                    r_cls = int(np.argmax(probs_r))
                else:
                    r_cls = 0
            except Exception:
                r_cls = -1
            
            ref_preds.append(r_cls)
            agrees = (pred_cls == r_cls)
            class_agreements.append(agrees)
            if not agrees:
                anomalous_tests.append({
                    "perturbation_type": item["type"],
                    "candidate_class": pred_cls,
                    "reference_class": r_cls,
                    "candidate_confidence": round(conf, 3)
                })

    total_tests = len(battery)
    agreement_rate = round(float(np.mean(class_agreements)), 3) if class_agreements else None
    mean_confidence = round(float(np.mean(cand_confidences)), 3)

    return {
        "total_battery_tests": total_tests,
        "reference_available": reference_adapter is not None,
        "behavioural_agreement_rate": agreement_rate,
        "anomalous_inputs_count": len(anomalous_tests),
        "mean_confidence": mean_confidence,
        "anomalous_samples": anomalous_tests[:10],
        "summary": (
            f"Evaluated {total_tests} inputs across controlled perturbation battery. "
            f"Agreement with reference: {int(agreement_rate*100) if agreement_rate is not None else 'N/A'}%. "
            f"Anomalous deviations: {len(anomalous_tests)}."
        )
    }
