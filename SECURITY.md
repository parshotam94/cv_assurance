# VisionTrust Assurance: Security Model & Cryptographic Specifications

## 1. Cryptographic Primitives

VisionTrust Assurance uses standard, high-performance cryptographic primitives implemented via the Python `cryptography` library:

| Function | Primitive / Standard | Implementation | Purpose |
|---|---|---|---|
| **Asymmetric Signatures** | Ed25519 (EdDSA over Curve25519) | `cryptography.hazmat.primitives.asymmetric.ed25519` | Inference provenance attestation |
| **Cryptographic Hashing** | SHA-256 (FIPS 180-4) | `hashlib.sha256` | Content addressing & audit hash-chaining |
| **Perceptual Hashing** | pHash (DCT 64-bit) & dHash (Gradient 64-bit) | `imagehash` | Image near-duplicate detection |
| **Entropy Generation** | CSPRNG | `secrets.token_hex` & `uuid.uuid4` | Nonce and session identifier generation |

---

## 2. Key Management & Air-Gapped Lifecycle

- **Key Generation**: Upon initialization (`scripts/initialize.py`), a 256-bit Ed25519 private key is generated locally using OS-level cryptographically secure random entropy.
- **Storage**: Keys are stored in the local file system at `data/keys/`:
  - `data/keys/ed25519_private.pem`: PKCS#8 encoded private key (read-only permissions).
  - `data/keys/ed25519_public.pem`: SubjectPublicKeyInfo encoded public key.
- **Air-Gapped Operation**: No keys, hashes, or telemetry are ever transmitted across networks. Digital signatures are verified locally against the public key extracted from PEM or embedded in canonical records.

---

## 3. Tamper-Evident Audit Ledger Design

The audit ledger maintains an unbroken cryptographic chain of custody:

### Block Structure
Every event recorded in the audit ledger contains:
- `event_id`: Unique identifier (`EVT-...`).
- `timestamp`: UTC ISO-8601 timestamp.
- `actor`: System identity or authenticated local user.
- `action`: Operation name (e.g. `DATASET_UPLOAD`, `MODEL_ANALYSIS`, `TAMPER_DETECTED`).
- `asset`: Target asset identifier (e.g. `Dataset:DS-01`, `Model:MOD-01`).
- `metadata_json`: Structured payload of the event.
- `previous_hash`: SHA-256 hash of the immediate predecessor event.
- `current_hash`: Calculated as:
$$\text{CurrentHash} = \text{SHA256}(\text{Timestamp} \,\|\, \text{Actor} \,\|\, \text{Action} \,\|\, \text{Asset} \,\|\, \text{MetadataCanonical} \,\|\, \text{PreviousHash})$$

### Genesis Invariant
The initial block in the database is created at system initialization with:
$$\text{PreviousHash}_0 = 0000000000000000000000000000000000000000000000000000000000000000$$

### Chain Verification Algorithm
During `/api/audit/verify`:
1. Events are retrieved in strict ascending sequential order.
2. The initial event must have `previous_hash == "0"*64`.
3. For every subsequent event $i$, its recorded `previous_hash` must equal $\text{CurrentHash}_{i-1}$.
4. $\text{CurrentHash}_i$ is independently re-computed from event fields and compared to the stored value.
5. If any mismatch occurs, the chain is declared **TAMPERED**, and the exact corrupted block index is returned.

---

## 4. Input Sanitization & Path Traversal Defense

All file uploads and system interactions adhere to strict input sanitization rules:
- **Filename Sanitization**: Uploaded filenames are stripped of path components (`../`, `..\\`), whitespace, and non-alphanumeric characters (preserving safe delimiters `_`, `-`, `.`).
- **Extension Whitelisting**: Strict validation of permitted file extensions:
  - Datasets: `{".zip", ".tar", ".gz"}`
  - Models: `{".onnx", ".pt", ".pth", ".ts"}`
  - Images: `{".jpg", ".jpeg", ".png", ".bmp", ".webp"}`
- **Isolation**: Extracted archives are restricted to isolated subdirectories within `data/datasets/<dataset_id>/extracted` to prevent directory traversal or file collisions.

---

## 5. Threat Model Assumptions

1. **Air-Gapped Host Integrity**: The local operating system and Python runtime are assumed to be trusted. Physical access to RAM allows key extraction; therefore, host-level disk encryption (e.g., BitLocker/LUKS) is recommended for production deployment.
2. **Untrusted Data Supply Chain**: All uploaded datasets, models, and runtime inference requests are treated as untrusted and potentially adversarial.
3. **Adversary Capabilities**: Adversaries may attempt to:
   - Inject poisoned samples into datasets.
   - Substitute trained models with Trojaned binaries.
   - Intercept and modify inference results in transit.
   - Replay legitimate historical inference results.
   - Alter local audit records to cover tracks.
