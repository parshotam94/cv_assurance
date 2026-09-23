/**
 * VisionTrust Assurance - Local API Client Module
 * 100% Offline / Local REST API Abstraction
 */

const API_BASE = '/api';

export const API = {
  // ---------------- Health & System ----------------
  async getHealth() {
    const res = await fetch(`${API_BASE}/health`);
    if (!res.ok) throw new Error(`Health check failed: ${res.statusText}`);
    return await res.json();
  },

  // ---------------- Datasets ----------------
  async listDatasets() {
    const res = await fetch(`${API_BASE}/datasets`);
    if (!res.ok) throw new Error(`Failed to list datasets: ${res.statusText}`);
    return await res.json();
  },

  async getDataset(datasetId) {
    const res = await fetch(`${API_BASE}/datasets/${datasetId}`);
    if (!res.ok) throw new Error(`Failed to get dataset: ${res.statusText}`);
    return await res.json();
  },

  async uploadDataset(file, datasetName) {
    const formData = new FormData();
    formData.append('file', file);
    if (datasetName) formData.append('dataset_name', datasetName);

    const res = await fetch(`${API_BASE}/datasets/upload`, {
      method: 'POST',
      body: formData,
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: res.statusText }));
      throw new Error(err.detail || 'Dataset upload failed');
    }
    return await res.json();
  },

  async analyzeDataset(datasetId, duplicateThreshold = 0.92, oodThreshold = 0.95) {
    const res = await fetch(
      `${API_BASE}/datasets/${datasetId}/analyze?duplicate_threshold=${duplicateThreshold}&ood_threshold=${oodThreshold}`,
      { method: 'POST' }
    );
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: res.statusText }));
      throw new Error(err.detail || 'Dataset analysis failed');
    }
    return await res.json();
  },

  async getDatasetSamples(datasetId, suspiciousOnly = false, limit = 50, offset = 0) {
    const res = await fetch(
      `${API_BASE}/datasets/${datasetId}/samples?suspicious_only=${suspiciousOnly}&limit=${limit}&offset=${offset}`
    );
    if (!res.ok) throw new Error(`Failed to load samples: ${res.statusText}`);
    return await res.json();
  },

  // ---------------- Models ----------------
  async listModels() {
    const res = await fetch(`${API_BASE}/models`);
    if (!res.ok) throw new Error(`Failed to list models: ${res.statusText}`);
    return await res.json();
  },

  async getModel(modelId) {
    const res = await fetch(`${API_BASE}/models/${modelId}`);
    if (!res.ok) throw new Error(`Failed to get model: ${res.statusText}`);
    return await res.json();
  },

  async uploadModel(file, modelName) {
    const formData = new FormData();
    formData.append('file', file);
    if (modelName) formData.append('model_name', modelName);

    const res = await fetch(`${API_BASE}/models/upload`, {
      method: 'POST',
      body: formData,
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: res.statusText }));
      throw new Error(err.detail || 'Model upload failed');
    }
    return await res.json();
  },

  async analyzeModel(modelId, referenceModelId = null, batterySamples = 60) {
    let url = `${API_BASE}/models/${modelId}/analyze?battery_samples=${batterySamples}`;
    if (referenceModelId) {
      url += `&reference_model_id=${encodeURIComponent(referenceModelId)}`;
    }
    const res = await fetch(url, { method: 'POST' });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: res.statusText }));
      throw new Error(err.detail || 'Model assurance analysis failed');
    }
    return await res.json();
  },

  // ---------------- Inference Provenance ----------------
  async runInference(file, modelId, preprocessingJson = null, configJson = null) {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('model_id', modelId);
    if (preprocessingJson) formData.append('preprocessing_json', preprocessingJson);
    if (configJson) formData.append('config_json', configJson);

    const res = await fetch(`${API_BASE}/inference/run`, {
      method: 'POST',
      body: formData,
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: res.statusText }));
      throw new Error(err.detail || 'Inference execution failed');
    }
    return await res.json();
  },

  async verifyInference(recordPayload) {
    const res = await fetch(`${API_BASE}/inference/verify`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(recordPayload),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: res.statusText }));
      throw new Error(err.detail || 'Inference verification failed');
    }
    return await res.json();
  },

  async simulateTampering(recordId, tamperField = 'output') {
    const formData = new FormData();
    formData.append('record_id', recordId);
    formData.append('tamper_field', tamperField);

    const res = await fetch(`${API_BASE}/inference/tamper-test`, {
      method: 'POST',
      body: formData,
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: res.statusText }));
      throw new Error(err.detail || 'Tamper simulation failed');
    }
    return await res.json();
  },

  async simulateReplay(originalRecordId) {
    const formData = new FormData();
    formData.append('original_record_id', originalRecordId);

    const res = await fetch(`${API_BASE}/inference/replay-test`, {
      method: 'POST',
      body: formData,
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: res.statusText }));
      throw new Error(err.detail || 'Replay simulation failed');
    }
    return await res.json();
  },

  async listInferenceRecords(limit = 50) {
    const res = await fetch(`${API_BASE}/inference/records?limit=${limit}`);
    if (!res.ok) throw new Error(`Failed to list inference records: ${res.statusText}`);
    return await res.json();
  },

  // ---------------- Distribution Shift ----------------
  async analyzeDistribution(referenceDatasetId, candidateDatasetId) {
    const formData = new FormData();
    formData.append('reference_dataset_id', referenceDatasetId);
    formData.append('candidate_dataset_id', candidateDatasetId);

    const res = await fetch(`${API_BASE}/distribution/analyze`, {
      method: 'POST',
      body: formData,
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: res.statusText }));
      throw new Error(err.detail || 'Distribution analysis failed');
    }
    return await res.json();
  },

  // ---------------- Findings ----------------
  async listFindings(assetId = null, category = null, severity = null) {
    const params = new URLSearchParams();
    if (assetId) params.append('asset_id', assetId);
    if (category) params.append('category', category);
    if (severity) params.append('severity', severity);

    const res = await fetch(`${API_BASE}/findings?${params.toString()}`);
    if (!res.ok) throw new Error(`Failed to list findings: ${res.statusText}`);
    return await res.json();
  },

  async getFinding(findingId) {
    const res = await fetch(`${API_BASE}/findings/${findingId}`);
    if (!res.ok) throw new Error(`Failed to get finding details: ${res.statusText}`);
    return await res.json();
  },

  // ---------------- Audit Trail ----------------
  async listAuditEvents(limit = 100) {
    const res = await fetch(`${API_BASE}/audit?limit=${limit}`);
    if (!res.ok) throw new Error(`Failed to fetch audit events: ${res.statusText}`);
    return await res.json();
  },

  async verifyAuditChain() {
    const res = await fetch(`${API_BASE}/audit/verify`, { method: 'POST' });
    if (!res.ok) throw new Error(`Audit chain verification request failed: ${res.statusText}`);
    return await res.json();
  },

  // ---------------- Reports ----------------
  async generateReport(datasetId = null, modelId = null, reportTitle = null) {
    const formData = new FormData();
    if (datasetId) formData.append('dataset_id', datasetId);
    if (modelId) formData.append('model_id', modelId);
    if (reportTitle) formData.append('report_title', reportTitle);

    const res = await fetch(`${API_BASE}/reports/generate`, {
      method: 'POST',
      body: formData,
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: res.statusText }));
      throw new Error(err.detail || 'Report generation failed');
    }
    return await res.json();
  },

  async listReports() {
    const res = await fetch(`${API_BASE}/reports`);
    if (!res.ok) throw new Error(`Failed to list reports: ${res.statusText}`);
    return await res.json();
  },

  async getReport(reportId) {
    const res = await fetch(`${API_BASE}/reports/${reportId}`);
    if (!res.ok) throw new Error(`Failed to get report: ${res.statusText}`);
    return await res.json();
  },

  // ---------------- Benchmark Demo ----------------
  async runDemoBenchmark() {
    const res = await fetch(`${API_BASE}/demo/run`, { method: 'POST' });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: res.statusText }));
      throw new Error(err.detail || 'Demo execution failed');
    }
    return await res.json();
  },
};

// ---------------- UI Toast Utility ----------------
export function showToast(message, type = 'info') {
  let container = document.getElementById('toast-container');
  if (!container) {
    container = document.createElement('div');
    container.id = 'toast-container';
    container.className = 'toast-container';
    document.body.appendChild(container);
  }

  const toast = document.createElement('div');
  toast.className = `toast toast-${type === 'error' ? 'error' : 'success'}`;
  toast.innerHTML = `
    <span>${type === 'error' ? '[!]' : '[+]'}</span>
    <div>${message}</div>
  `;
  container.appendChild(toast);

  setTimeout(() => {
    toast.style.opacity = '0';
    setTimeout(() => toast.remove(), 300);
  }, 4000);
}

// ---------------- Risk Color Utility ----------------
export function getRiskBadge(severityOrLevel) {
  const norm = (severityOrLevel || 'LOW').toUpperCase();
  switch (norm) {
    case 'CRITICAL':
    case 'QUARANTINE':
      return `<span class="badge badge-critical">${norm}</span>`;
    case 'HIGH':
    case 'REVIEW':
      return `<span class="badge badge-high">${norm}</span>`;
    case 'MEDIUM':
      return `<span class="badge badge-med">${norm}</span>`;
    default:
      return `<span class="badge badge-low">${norm}</span>`;
  }
}
