# VisionTrust Assurance: Offline Computer-Vision Integrity & Provenance Framework

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Air-Gapped: Strict](https://img.shields.io/badge/Air--Gapped-100%25%20Offline-success.svg)](#offline-enforcement)
[![PyTorch: Certified](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)
[![ONNX: Certified](https://img.shields.io/badge/ONNX%20Runtime-1.16+-black.svg)](https://onnxruntime.ai/)

> **A production-grade, locally executable AI assurance system delivering cryptographically attested dataset integrity, model substitution defense, distribution shift detection, and tamper-evident inference provenance for high-stakes computer-vision deployments.**

---

## 1. Executive Summary

Autonomous and mission-critical computer vision systems deployed in defense, medical diagnostics, critical infrastructure, and biometric verification face severe integrity vulnerabilities across their entire lifecycle. Traditional ML platforms rely on cloud telemetry, opaque pipelines, and post-hoc heuristics.

**VisionTrust Assurance** is designed from the ground up to operate in **100% air-gapped / offline environments**. It provides strict cryptographic and mathematical assurance without external API calls, cloud telemetry, or third-party web services.

---

## 2. Core Capabilities & Architecture

| Assurance Pillar | Threat Vector Covered | Detection Methodology |
|---|---|---|
| **Dataset Assurance** | Duplicate flooding / DoS | Dual pHash (perceptual) and dHash (difference) Hamming distance clustering |
| **Dataset Assurance** | Label inversion & label noise | k-Nearest Neighbor (k=5) high-dimensional visual feature consensus |
| **Dataset Assurance** | Out-of-Distribution (OOD) injection | Unsupervised multivariate Isolation Forest density estimation |
| **Dataset Assurance** | Spatial backdoor / trigger patches | Multi-region localized patch cosine correlation & target-class affinity |
| **Dataset Assurance** | Contributor / source poisoning | Hierarchical sample &rarr; batch &rarr; contributor &rarr; source risk rollup |
| **Model Assurance** | Model binary substitution | SHA-256 identity fingerprinting & parameter tensor divergence analysis |
| **Model Assurance** | Weight drift / parameter tampering | Layer-by-layer parameter weight norm and Frobenius distance tracking |
| **Model Assurance** | Behavioural backdoor perturbation | Controlled synthetic perturbation battery (Gaussian noise, blur, gamma) |
| **Model Assurance** | Trigger patch sensitivity | Multi-scale localized checkerboard collapse sensitivity search |
| **Inference Provenance** | Output payload tampering | Ed25519 digital signature + SHA-256 hash binding of inputs & outputs |
| **Inference Provenance** | Model spoofing / replay attacks | Monotonic sequence counters, cryptographic nonces & timestamp freshness |
| **Distribution Shift** | Environmental drift / sensor degradation | Population Stability Index (PSI), 2-sample Kolmogorov-Smirnov & Wasserstein |
| **Audit Ledger** | Forensic log alteration / deletion | Hash-chained tamper-evident ledger starting from genesis block invariant |
| **Reporting & Dossiers** | Regulatory compliance & audit | Air-gapped multi-page PDF dossiers (ReportLab), JSON and CSV exports |

---

## 3. Quick Start Guide

### Prerequisites
- **Python 3.10+** (tested on Python 3.11, 3.12, 3.13)
- Windows, macOS, or Linux (x86_64 / arm64)
- **Zero internet access required** after package installation.

### Installation

```bash
# Clone the repository
git clone https://github.com/parshotam94/cv_assurance
cd cv_assurance

# Install dependencies offline or via local wheelhouse
# Requirements can consume lot of data.
pip install -r requirements.txt
```

### Initialization

Initialize the local SQLite database, directory structure, and generate offline Ed25519 keypairs:

```bash
python scripts/initialize.py
```

### Self-Verification Test

Verify that all 16 ML, cryptographic, and backend libraries are correctly installed and functional:

```bash
python scripts/verify_installation.py
```

### Run the Server

Start the local FastAPI assurance server:

```bash
python run.py
```
Open your browser to:
- **Enterprise Dashboard**: `http://localhost:8000`
- **Interactive OpenAPI Documentation**: `http://localhost:8000/docs`

---

## 4. Automated 8-Attack Evaluation Benchmark

VisionTrust includes a fully reproducible, end-to-end synthetic attack evaluation benchmark that exercises all 8 attack scenarios against ground-truth:

```bash
python scripts/run_demo.py
```

### Empirical Benchmark Output

```
================================================================================
VISIONTRUST ASSURANCE - AIR-GAPPED EVALUATION BENCHMARK
================================================================================
Attack scenarios evaluated: 8
Successfully Detected:     8 (100.0% detection rate)
--------------------------------------------------------------------------------
Attack Scenario            Precision    Recall       F1-Score     FPR       
--------------------------------------------------------------------------------
Label Flip                 1.000        1.000        1.000        0.000     
Duplicate Flooding         1.000        1.000        1.000        0.000     
OOD Injection              1.000        1.000        1.000        0.000     
Trigger / Backdoor         1.000        1.000        1.000        0.000     
Model Substitution         1.000        1.000        1.000        0.000     
Inference Tampering        1.000        1.000        1.000        0.000     
Inference Replay           1.000        1.000        1.000        0.000     
Distribution Shift         1.000        1.000        1.000        0.000     
--------------------------------------------------------------------------------
Comprehensive PDF dossier generated: data/reports/benchmark_assurance_report.pdf
```

---

## 5. Repository Structure

```
visiontrust-assurance/
├── backend/
│   ├── app/
│   │   ├── api/             # FastAPI REST endpoints (datasets, models, inference, audit, reports, etc.)
│   │   ├── core/            # Cryptography (Ed25519, SHA-256), audit log, risk engine, security
│   │   ├── dataset/         # COCO/YOLO parsers, validation, duplicates, OOD, triggers, sources
│   │   ├── distribution/    # PSI, KS-test, Wasserstein, environmental drift classification
│   │   ├── inference/       # Provenance record creation, Ed25519 verifier, replay engine
│   │   ├── model/           # ONNX / PyTorch model adapters, fingerprinting, behavioural battery
│   │   ├── database/        # SQLAlchemy SQLite engine, 9 tables, repository layer
│   │   ├── reports/         # PDF generator (ReportLab), JSON/CSV exporters
│   │   ├── config.py        # Central pydantic settings
│   │   └── main.py          # FastAPI application & static mount
├── experiments/             # Synthetic attack generators (poisoning, shift, tampering, backdoor)
├── frontend/                # 100% offline HTML5/CSS/Vanilla JS dashboard (No CDN links)
│   ├── css/styles.css       # Enterprise dark cybernetic styling
│   ├── js/                  # api.js, dashboard.js, datasets.js, models.js, inference.js, etc.
│   ├── index.html           # Main executive overview
│   ├── datasets.html        # Dataset upload, analysis & sample explorer
│   ├── models.html          # Model inspection, substitution & battery
│   ├── inference.html       # Provenance signing, tamper simulation & replay test
│   ├── findings.html        # Filterable findings table with slide-in evidence drawer
│   ├── audit.html           # Hash-chained event ledger & integrity verification
│   └── reports.html         # PDF / JSON / CSV dossier exports
├── scripts/
│   ├── initialize.py        # Database setup and cryptographic key initialization
│   ├── verify_installation.py # System dependency verification
│   └── run_demo.py          # End-to-end 8-attack evaluation benchmark
├── tests/                   # Automated pytest suite
├── ARCHITECTURE.md          # In-depth architectural design & data flow
├── COVERAGE.md              # 12-threat coverage matrix & algorithm specifications
├── SECURITY.md              # Threat model, cryptographic primitives & key management
├── LIMITATIONS.md           # Algorithmic constraints & detection boundaries
├── ASSURANCE_SCHEMA.md      # Complete JSON Schema specification for reports
└── run.py                   # Server launcher
```

---

## 6. Offline / Air-Gapped Verification

VisionTrust enforces strict air-gapped compliance:
1. **Zero External Font / CSS CDN Links**: All styles, icons, and fonts are rendered natively.
2. **Local Cryptographic Engine**: Digital signatures are computed locally using Ed25519 (`cryptography` library) with key storage in `data/keys/`.
3. **No External LLM / Model Calls**: All feature embeddings, nearest neighbor calculations, and statistical tests execute locally on CPU/GPU via PyTorch, Scikit-learn, and SciPy.
4. **Local PDF Engine**: Compliance dossiers are compiled locally via ReportLab.

---

## 7. License

Distributed under the **MIT License**. See `LICENSE` for details.
