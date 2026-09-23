# VisionTrust Assurance: Threat Coverage Matrix (12 Pillars)

VisionTrust Assurance provides multi-layer protection across 12 distinct threat vectors spanning the Computer Vision lifecycle.

---

## 1. Threat Coverage Overview

| ID | Threat Vector | Life Cycle Phase | Algorithmic Detection Strategy | Target Confidence |
|---|---|---|---|---|
| **THR-01** | Near-Duplicate Flooding | Ingestion / Curation | Dual pHash / dHash Hamming clustering & cluster size ratio | 95% |
| **THR-02** | Systematic Label Inversion | Annotation / Curation | k-NN visual feature consensus & class boundary violation | 85% |
| **THR-03** | Out-of-Distribution Infiltration | Data Ingestion | Multivariate Isolation Forest on color & spatial texture | 90% |
| **THR-04** | Spatial Backdoor / Trigger Patch | Data Ingestion | Multi-scale sliding window patch cosine correlation & target class skew | 85% |
| **THR-05** | Contributor / Source Poisoning | Data Supply Chain | Hierarchical source/batch anomaly rollup | 90% |
| **THR-06** | Model Binary Substitution | Model Deployment | Cryptographic SHA-256 fingerprinting & tensor parameter comparison | 100% |
| **THR-07** | Weight Modification / Drift | Model Fine-tuning | Frobenius weight norms & parameter difference ratio | 95% |
| **THR-08** | Behavioural Perturbation Vulnerability | Model Evaluation | Synthetic environmental stress battery (noise, blur, contrast) | 90% |
| **THR-09** | Model Trigger Sensitivity | Model Evaluation | Targeted localized checkerboard perturbation search | 85% |
| **THR-10** | Inference Output Tampering | Runtime Inference | Ed25519 digital signature & canonical SHA-256 hash binding | 100% |
| **THR-11** | Inference Replay Attacks | Runtime Inference | Monotonic sequence counters & cryptographic nonce validation | 100% |
| **THR-12** | Environmental & Sensor Shift | Operational Monitoring | Population Stability Index (PSI), 2-sample KS-test & Wasserstein | 92% |

---

## 2. In-Depth Threat Specifications

### THR-01: Near-Duplicate Flooding
- **Attack Vector**: An adversary floods the training set with slightly modified copies of specific scenes to bias class priors or execute a Denial-of-Service on training compute.
- **Detection Method**: Every image is resized to 32x32 grayscale. A 2D Discrete Cosine Transform (DCT) extracts the top 8x8 low-frequency components to produce a 64-bit `pHash`. A row/column gradient produces a 64-bit `dHash`. If $\text{HammingDist}(H_1, H_2) \le 5$, images are clustered. Clusters exceeding 10% of total records are classified as Flooding attacks.
- **Disposition**: `REVIEW` (Low cluster counts) or `QUARANTINE` (Flooding).

### THR-02: Systematic Label Inversion
- **Attack Vector**: Intentional or accidental mislabeling of training samples to degrade decision boundaries (e.g. labeling pedestrians as background).
- **Detection Method**: Visual feature embeddings (color histograms + edge gradients) are projected into metric space. For each sample, the $k=5$ nearest visual neighbors are queried. If $\ge 80\%$ of neighbors share a conflicting label, the sample is flagged.
- **Disposition**: `REVIEW` if $< 5$ samples, `QUARANTINE` if $\ge 5$ samples.

### THR-03: Out-of-Distribution Infiltration
- **Attack Vector**: Submitting images from irrelevant domains, corrupted sensors, or synthetic adversarial noise distributions.
- **Detection Method**: Multivariate statistical feature vectors (RGB channel means, variances, skewness, sharpness, high-frequency energy) are modeled with an unsupervised Isolation Forest. Samples falling beyond the 95th anomaly percentile are flagged.
- **Disposition**: `REVIEW` if $< 10$ samples, `QUARANTINE` if $\ge 10$ samples.

### THR-04: Spatial Backdoor / Trigger Patch
- **Attack Vector**: Injecting localized visual watermarks (e.g. checkerboard patches) into training images to train a dormant backdoor trigger.
- **Detection Method**: Sliding window (16x16 / 32x32) extracts candidate patches across high-gradient areas. Patches with cosine similarity $\ge 0.85$ appearing across multiple images are tested for label correlation. If $>75\%$ of matched images share the same class, a trigger backdoor finding is emitted.
- **Disposition**: `QUARANTINE`.

### THR-05: Contributor / Source Poisoning
- **Attack Vector**: A compromised vendor, rogue annotator, or malicious batch ingestion pipeline infusing poisoned samples.
- **Detection Method**: Samples are grouped by `contributor_id`, `batch_id`, and `source_id`. The anomaly rate per entity is calculated:
$$\text{Rate} = \frac{N_{\text{dup}} + N_{\text{mislabel}} + N_{\text{ood}} + N_{\text{trigger}}}{N_{\text{total\_source}}}$$
Sources with anomaly rate $> 25\%$ are flagged as compromised.
- **Disposition**: `QUARANTINE` source.

### THR-06 & THR-07: Model Substitution & Weight Modification
- **Attack Vector**: A rogue employee or supply-chain attacker swaps a verified model binary with an unverified or trojaned version.
- **Detection Method**:
  1. Computes binary SHA-256. If reference model exists and hashes differ, triggers substitution analysis.
  2. Inspects layer names, tensor shapes, and parameter counts.
  3. Computes parameter difference ratio and relative Frobenius norm deviation.
- **Disposition**: `QUARANTINE` if parameters differ or norm difference $> 20\%$.

### THR-08 & THR-09: Behavioural Battery & Trigger Sensitivity
- **Attack Vector**: A model with hidden backdoors performs normally on standard test sets but fails under specific conditions or triggers.
- **Detection Method**: Runs candidate model through a 4-parameter perturbation battery (Gaussian noise, Gaussian blur, illumination shifts, contrast stretching). If prediction agreement with reference falls below 85%, behavioral divergence is flagged. Also tests vulnerability to localized patch collapse.
- **Disposition**: `REVIEW` or `QUARANTINE`.

### THR-10 & THR-11: Inference Tampering & Replay Attacks
- **Attack Vector**: An attacker in the network modifies inference outputs (e.g., changing "weapon" to "harmless") or replays previously captured valid inference sessions.
- **Detection Method**:
  1. Output payloads, model identity, input image, and configs are hashed and verified against the Ed25519 digital signature.
  2. Monotonic sequence numbers and high-entropy nonces are tracked in a persistent SQLite ledger. Any sequence regression or reused nonce triggers an immediate Replay attack finding.
- **Disposition**: `QUARANTINE` and immediate audit alert.

### THR-12: Environmental & Distribution Shift
- **Attack Vector**: Gradual camera lens degradation, weather variations (fog, night, rain), or seasonal lighting shifts causing silent model degradation.
- **Detection Method**: Computes Population Stability Index (PSI) across 8 visual attributes:
$$\text{PSI} = \sum_{b=1}^{B} (P_b - Q_b) \ln\left(\frac{P_b}{Q_b}\right)$$
Performs 2-sample Kolmogorov-Smirnov test and Wasserstein distance. Categorizes shifts into `NORMAL`, `OPERATIONAL_DRIFT`, `ANOMALOUS`, or `SUSPICIOUS`.
- **Disposition**: `ACCEPT` (Normal), `REVIEW` (Operational Drift), or `QUARANTINE` (Suspicious).
