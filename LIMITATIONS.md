# VisionTrust Assurance: Technical Limitations & Detection Boundaries

To maintain scientific integrity and prevent false sense of security, this document outlines the explicit mathematical, architectural, and operational boundaries of the VisionTrust Assurance framework.

---

## 1. Algorithmic Limitations

### 1.1 Perceptual Hashing (Duplicates & Flooding)
- **Scale and Rotation Invariance**: Standard 64-bit `pHash` and `dHash` operate on 32x32 DCT representations. While highly resilient to minor compression artifacts, brightness changes, and small translations, they do **not** reliably detect heavy rotations ($> 15^\circ$), extreme affine shearing, or severe non-linear crops.
- **Collisions in Low-Information Images**: Solid black, solid white, or single-color background images produce identical perceptual hashes and will be clustered together even if semantically distinct.

### 1.2 Feature-Space Nearest Neighbors (Label Mislabelling)
- **Low-Level Statistical Embeddings**: The default feature extractor relies on 64-dimensional statistical color moments and spatial edge gradients to ensure 100% offline, lightweight execution without large neural backbones. Consequently, fine-grained semantic distinctions (e.g. distinguishing two visually identical animal breeds) may exhibit feature overlap and yield false mislabeling alarms.
- **Small Population Effects**: In datasets with $< 20$ samples per class, the $k=5$ neighborhood consensus may have insufficient sample density to establish robust class boundaries.

### 1.3 Unsupervised Out-of-Distribution (Isolation Forest)
- **Manifold Assumption**: In the absence of an explicit reference baseline dataset, the Isolation Forest evaluates anomaly scores relative to the internal distribution of the candidate dataset itself. If an entire dataset consists of uniformly corrupted images, the detector will treat that corruption as normal manifold density.
- **Curse of Dimensionality**: High-dimensional feature spaces require adequate sample counts to prevent sparse density estimates.

### 1.4 Spatial Patch Backdoors (Trigger Search)
- **Fixed Patch Assumption**: The trigger search engine scans for fixed or low-variance spatial patches (such as checkerboards, stickers, or high-contrast stamps). It does **not** detect:
  - Steganographic, imperceptible $L_\infty$ perturbations ($\epsilon \le 4/255$).
  - Dynamic, sample-specific generative triggers.
  - Blended watermarks across the entire image canvas.
  - Frequency-domain (invisible Fourier) perturbations.

### 1.5 Model Substitution & Behavioural Battery
- **Baseline Dependency**: Detection of model substitution and weight drift requires a verified reference baseline model. Standalone candidate models can be fingerprinted for identity, but behavioral drift cannot be asserted without a baseline reference.
- **Black-Box vs. White-Box Trade-Off**: For ONNX models (gray-box), internal layer activations and gradient maps cannot be hooked directly. Only PyTorch (`TorchModelAdapter`) models support white-box layer-level activation hooking.

---

## 2. Operational & Platform Boundaries

- **Single-Node Storage**: VisionTrust Assurance currently utilizes a local SQLite database for zero-configuration, air-gapped deployment. While ideal for edge deployments and local appliances, SQLite is limited to single-node concurrent write transactions.
- **Upload Constraints**: By default, dataset archives and model binaries are capped at 500 MB per request to preserve system memory on edge hardware.
- **Adversarial Inference-Time Attacks**: Provenance attestation guarantees that an inference output was genuinely produced by the attested model from the given input image. It does **not** guarantee that the input image was not an adversarial example designed to fool the underlying model (e.g., FGSM or PGD attacks).
