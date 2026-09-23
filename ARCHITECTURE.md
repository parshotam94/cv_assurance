# VisionTrust Assurance: System Architecture & Technical Specifications

## 1. High-Level Architecture Overview

VisionTrust Assurance is structured as a modular, offline-first security and assurance framework for Computer Vision (CV) pipelines. The architecture separates concern into five distinct layers:

```
+-----------------------------------------------------------------------------------+
|                           PRESENTATION LAYER (OFFLINE)                            |
|  - Enterprise Dark Dashboard (HTML5 / Vanilla JS / Local CSS)                     |
|  - Dataset Explorer, Model Inspector, Provenance Verifier, Audit Ledger           |
+-----------------------------------------------------------------------------------+
                                         │  (REST API calls)
+-----------------------------------------------------------------------------------+
|                            APPLICATION LAYER (FastAPI)                            |
|  - Routers: /datasets, /models, /inference, /distribution, /findings, /audit, etc. |
|  - Asynchronous background tasking & file handling                                |
+-----------------------------------------------------------------------------------+
        │                         │                           │
+─────────────────+      +─────────────────+         +─────────────────+
| DATASET ENGINES |      |  MODEL ENGINES  |         | PROVENANCE CORE |
| - Dual pHash/dHash     | - ONNX/Torch    |         | - Ed25519 Sign  |
| - k-NN Labels   |      |   Adapters      |         | - SHA-256 Hashes|
| - Isolation     |      | - Fingerprint   |         | - Canonical JSON|
|   Forest OOD    |      | - Perturbation  |         | - Monotonic Seq |
| - Patch Triggers|      |   Battery       |         | - Replay Store  |
| - Source Rollup |      | - Trigger Sens. |         +─────────────────+
+─────────────────+      +─────────────────+                  │
        │                         │                           │
+───────────────────────────────────────────────────────────────────────────────────+
|                            CRYPTOGRAPHIC & RISK CORE                              |
|  - SHA-256 Content-Addressable Storage & Tensors                                  |
|  - Hash-Chained Audit Ledger with Genesis Block Verification                      |
|  - Weighted Multi-Evidence Risk Scoring Engine                                    |
+───────────────────────────────────────────────────────────────────────────────────+
                                         │
+───────────────────────────────────────────────────────────────────────────────────+
|                             DATA & PERSISTENCE LAYER                              |
|  - SQLite Database (9 Relational Tables)                                          |
|  - Local File Store: data/{datasets, models, inference, reports, keys}            |
+-----------------------------------------------------------------------------------+
```

---

## 2. Core Subsystems

### 2.1 Dataset Assurance Engine
- **Perceptual Duplication**: Computes 64-bit perceptual hash (`pHash`) and difference hash (`dHash`) using Discrete Cosine Transform (DCT) over grayscale normalized images. Near-duplicates are clustered using union-find with Hamming distance threshold $\le 5$ (similarity $\ge 0.92$). Duplicate flooding is flagged when cluster size exceeds $10\%$ of total dataset samples.
- **Label Integrity**: Extracts compact 64-dimensional color and spatial texture features per sample. Runs k-Nearest Neighbors ($k=5$) across the dataset. Samples whose ground-truth label conflicts with $\ge 80\%$ of their visual neighbors are flagged for label noise or inversion.
- **Out-of-Distribution (OOD)**: Trains an unsupervised multivariate Isolation Forest on low-level statistical moments (mean, variance, skewness, sharpness, and high-frequency edge energy). Scores above the 95th percentile are flagged as distribution outliers.
- **Trigger / Backdoor Detection**: Performs sliding-window local patch cosine similarity analysis across images. Identifies spatially localized patterns with high visual similarity that disproportionately correlate with specific class assignments.
- **Hierarchical Source Attribution**: Rolls up sample-level anomalies across batch IDs and data contributors to compute source risk:
$$\text{Risk}_{\text{source}} = \max\left(\frac{N_{\text{anomalous}}}{N_{\text{samples}}}, \dots\right)$$

### 2.2 Model Assurance Engine
- **Model Adapters**:
  - `ONNXModelAdapter`: Gray-box analysis using `onnxruntime` and ONNX graph topology parsing.
  - `TorchModelAdapter`: White-box analysis using PyTorch tensor inspection, forward hooks, and layer activation analysis.
- **Cryptographic Fingerprinting**: Computes SHA-256 of the model binary, extracts parameter counts, layer topologies, and computes weight tensor Frobenius norms to detect fine-tuning, quantization drift, or substitution.
- **Controlled Perturbation Battery**: Tests candidate models against synthetic environmental perturbations:
  - Gaussian blur ($\sigma \in [1.0, 3.0]$)
  - Gaussian additive noise ($\sigma \in [0.05, 0.2]$)
  - Contrast & brightness variations ($\pm 40\%$)
  - Compares prediction drift against reference baseline.
- **Trigger Sensitivity**: Injects synthetic localized checkerboard trigger patches at varying image coordinates to evaluate target-class collapse probability.

### 2.3 Cryptographic Inference Provenance
Every inference output is bound into a cryptographically signed canonical record:
1. `input_sha256`: SHA-256 hash of raw input image bytes.
2. `model_sha256`: SHA-256 identity of the registered model binary.
3. `preprocessing_sha256`: Canonical JSON hash of resize dimensions, normalization means, and color space.
4. `config_sha256`: Canonical JSON hash of inference threshold, batch size, and execution provider.
5. `output_sha256`: Canonical JSON hash of prediction classes, confidence scores, and raw logits.
6. `record_hash`: SHA-256 of canonical payload string:
$$\text{Payload} = \text{input\_sha256} \,\|\, \text{model\_sha256} \,\|\, \text{output\_sha256} \,\|\, \text{timestamp} \,\|\, \text{nonce} \,\|\, \text{sequence}$$
7. `signature`: Ed25519 digital signature over `record_hash` using local private key.
8. `replay_check`: Validates sequence monotonicity ($Seq_{new} > Seq_{latest}$) and nonce uniqueness.

### 2.4 Hash-Chained Audit Ledger
All security-relevant actions (uploads, analyses, verifications, tampering events) are appended to a hash-chained ledger:
$$\text{Hash}_i = \text{SHA256}(\text{Timestamp}_i \,\|\, \text{Actor}_i \,\|\, \text{Action}_i \,\|\, \text{Asset}_i \,\|\, \text{Metadata}_i \,\|\, \text{Hash}_{i-1})$$
- Genesis block constraint: $\text{Hash}_0 = 0^{64}$.
- Chain verification executes backwards and forwards to detect deletions, modifications, or invalid links.

### 2.5 Multi-Evidence Risk Engine
Overall risk scores ($0.0 \dots 100.0$) are computed from findings using category-weighted severity:
$$\text{Score} = \min\left(100.0, \, \sum_{i=1}^{M} w_{\text{cat}} \cdot w_{\text{sev}} \cdot \text{Conf}_i \cdot \sqrt{\frac{N_{\text{affected}}}{N_{\text{total}}}}\right)$$
- $\text{Score} < 40.0$: `ACCEPT` (Green)
- $40.0 \le \text{Score} < 75.0$: `REVIEW` (Yellow/Orange)
- $\text{Score} \ge 75.0$: `QUARANTINE` (Red)

---

## 3. Database Schema (9 Relational Tables)

1. `datasets`: ID, name, format (COCO/YOLO), file path, SHA-256, sample count, risk score, disposition.
2. `dataset_samples`: Sample ID, dataset ID, file path, labels, annotations, contributor, batch ID, suspicious flag.
3. `dataset_sources`: Source ID, dataset ID, sample count, suspicious count, risk score, disposition.
4. `models`: ID, name, framework, file path, SHA-256, parameter count, access level, risk score, disposition.
5. `inference_records`: Record ID, input hash, model hash, output hash, timestamp, nonce, sequence, signature, verification status.
6. `findings`: Finding ID, asset ID, asset type, category, severity, confidence, risk score, title, evidence, recommendations.
7. `audit_events`: Event ID, timestamp, actor, action, asset, previous hash, current hash, metadata JSON.
8. `assurance_reports`: Report ID, title, risk score, disposition, executive summary, PDF/JSON/CSV paths.
9. `experiments`: Benchmark scenario name, config, ground truth JSON, results JSON, precision, recall, F1.
