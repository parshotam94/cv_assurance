/**
 * VisionTrust Assurance - Datasets Page Logic
 */
import { API, showToast, getRiskBadge } from './api.js';

let currentDatasetList = [];

document.addEventListener('DOMContentLoaded', async () => {
  await loadDatasets();

  // Setup file dropzone
  const fileInput = document.getElementById('dataset-file-input');
  const filenameLabel = document.getElementById('dropzone-filename');
  if (fileInput && filenameLabel) {
    fileInput.addEventListener('change', (e) => {
      if (fileInput.files.length > 0) {
        filenameLabel.textContent = fileInput.files[0].name;
      }
    });
  }

  // Upload Form
  const form = document.getElementById('upload-dataset-form');
  if (form) {
    form.addEventListener('submit', async (e) => {
      e.preventDefault();
      const file = fileInput.files[0];
      if (!file) {
        showToast('Please select a dataset archive (.zip)', 'error');
        return;
      }
      const name = document.getElementById('dataset-name-input').value;
      const btn = document.getElementById('btn-upload-dataset');
      btn.disabled = true;
      btn.innerHTML = '<span class="spinner"></span> Ingesting...';

      try {
        const res = await API.uploadDataset(file, name);
        showToast(`Dataset ${res.dataset_id} uploaded successfully! (${res.sample_count} samples)`, 'success');
        form.reset();
        filenameLabel.textContent = 'Drag & drop dataset archive or click to select';
        await loadDatasets();
      } catch (err) {
        showToast(`Upload error: ${err.message}`, 'error');
      } finally {
        btn.disabled = false;
        btn.textContent = 'Upload & Ingest Archive';
      }
    });
  }

  // Sample explorer dataset selector
  const dsSelect = document.getElementById('select-dataset-samples');
  const chkSuspicious = document.getElementById('chk-suspicious-only');
  if (dsSelect) {
    dsSelect.addEventListener('change', () => loadSamples());
  }
  if (chkSuspicious) {
    chkSuspicious.addEventListener('change', () => loadSamples());
  }
});

async function loadDatasets() {
  try {
    const datasets = await API.listDatasets();
    currentDatasetList = datasets;
    const tbody = document.getElementById('datasets-tbody');
    const select = document.getElementById('select-dataset-samples');

    if (select) {
      const prevVal = select.value;
      select.innerHTML = '<option value="">Select Dataset...</option>' + 
        datasets.map(d => `<option value="${d.id}" ${d.id === prevVal ? 'selected' : ''}>${d.name} (${d.id})</option>`).join('');
    }

    if (!tbody) return;

    if (!datasets || datasets.length === 0) {
      tbody.innerHTML = `<tr><td colspan="8" style="text-align: center; color: var(--text-dim); padding: 2rem;">No datasets registered. Upload an archive above.</td></tr>`;
      return;
    }

    tbody.innerHTML = datasets.map(d => `
      <tr>
        <td><code>${d.id}</code></td>
        <td><strong>${d.name}</strong></td>
        <td><span class="badge badge-info">${d.format}</span></td>
        <td>${d.sample_count}</td>
        <td><span class="badge ${d.status === 'ANALYZED' ? 'badge-low' : 'badge-med'}">${d.status}</span></td>
        <td><strong>${d.risk_score ? d.risk_score.toFixed(1) : '0.0'}</strong></td>
        <td>${getRiskBadge(d.disposition)}</td>
        <td>
          <button class="btn btn-primary btn-sm btn-analyze" data-id="${d.id}">
            Analyze
          </button>
        </td>
      </tr>
    `).join('');

    // Attach analyze listeners
    document.querySelectorAll('.btn-analyze').forEach(btn => {
      btn.addEventListener('click', async (e) => {
        const id = e.target.getAttribute('data-id');
        await triggerAnalysis(id, e.target);
      });
    });

    // Automatically display analysis results & load samples for first analyzed dataset on load
    if (select && (!select.value || select.value === '')) {
      const analyzed = datasets.find(d => d.status === 'ANALYZED' && (d.risk_score > 0 || d.sample_count > 0));
      if (analyzed) {
        select.value = analyzed.id;
        try {
          const details = await API.getDataset(analyzed.id);
          displayAnalysisResults(details, false);
          await loadSamples();
        } catch (e) {
          console.warn('Could not auto-display dataset results:', e);
        }
      }
    }

  } catch (err) {
    showToast(`Failed to load datasets: ${err.message}`, 'error');
  }
}

async function triggerAnalysis(datasetId, buttonEl) {
  buttonEl.disabled = true;
  buttonEl.innerHTML = '<span class="spinner"></span> Analyzing...';

  try {
    showToast(`Running integrity engines on ${datasetId}...`, 'info');
    const res = await API.analyzeDataset(datasetId);
    showToast(`Analysis completed for ${datasetId}! Risk score: ${res.overall_risk_score.toFixed(1)}`, 'success');
    
    // Display results panel
    displayAnalysisResults(res, true);
    await loadDatasets();

    // Select in sample explorer
    const select = document.getElementById('select-dataset-samples');
    if (select) {
      select.value = datasetId;
      await loadSamples();
    }
  } catch (err) {
    showToast(`Analysis error: ${err.message}`, 'error');
  } finally {
    buttonEl.disabled = false;
    buttonEl.textContent = 'Analyze';
  }
}

function displayAnalysisResults(res, shouldScroll = true) {
  const panel = document.getElementById('dataset-analysis-panel');
  if (!panel) return;

  panel.style.display = 'block';
  const dsId = res.dataset_id || res.id || '';
  document.getElementById('analysis-ds-id').textContent = dsId;
  document.getElementById('analysis-ds-disposition').innerHTML = getRiskBadge(res.disposition || 'ACCEPT');

  const dupCount = res.duplicate_clusters !== undefined ? res.duplicate_clusters : (res.summary?.duplicate_clusters || 0);
  const mislabelCount = res.mislabelled_samples !== undefined ? res.mislabelled_samples : (res.summary?.mislabelled_count || 0);
  const oodCount = res.ood_samples !== undefined ? res.ood_samples : (res.summary?.ood_count || 0);
  const triggerCount = res.trigger_clusters !== undefined ? res.trigger_clusters : (res.summary?.trigger_clusters || 0);

  document.getElementById('res-dup-count').textContent = dupCount;
  document.getElementById('res-mislabel-count').textContent = mislabelCount;
  document.getElementById('res-ood-count').textContent = oodCount;
  document.getElementById('res-trigger-count').textContent = triggerCount;

  const sourcesDiv = document.getElementById('sources-breakdown-list');
  const sourceList = res.source_profiles || res.sources || [];
  if (sourcesDiv && sourceList.length > 0) {
    sourcesDiv.innerHTML = sourceList.map(s => `
      <div style="background: var(--bg-input); padding: 1rem; border-radius: 6px; border: 1px solid var(--border-color);">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
          <strong style="font-size: 0.85rem;">${s.source_id}</strong>
          ${getRiskBadge(s.risk_level || (s.risk_score >= 80 ? 'CRITICAL' : s.risk_score >= 60 ? 'HIGH' : s.risk_score >= 30 ? 'MEDIUM' : 'LOW'))}
        </div>
        <div style="font-size: 0.78rem; color: var(--text-dim);">
          <div>Samples: ${s.sample_count || s.samples || 0} (Suspicious: ${s.suspicious_count || s.suspicious || 0})</div>
          <div>Risk Score: ${(s.risk_score || 0).toFixed(1)}</div>
          <div>Disposition: <strong>${s.disposition || 'ACCEPT'}</strong></div>
        </div>
      </div>
    `).join('');
  }

  if (shouldScroll) {
    panel.scrollIntoView({ behavior: 'smooth' });
  }
}

async function loadSamples() {
  const select = document.getElementById('select-dataset-samples');
  const chkSuspicious = document.getElementById('chk-suspicious-only');
  const grid = document.getElementById('samples-grid');
  if (!select || !grid) return;

  const datasetId = select.value;
  if (!datasetId) {
    grid.innerHTML = `<div style="grid-column: 1 / -1; text-align: center; color: var(--text-dim); padding: 2rem;">Select a dataset above to inspect its visual samples.</div>`;
    return;
  }

  const suspiciousOnly = chkSuspicious ? chkSuspicious.checked : false;
  grid.innerHTML = `<div style="grid-column: 1 / -1; text-align: center; color: var(--text-dim); padding: 2rem;"><span class="spinner"></span> Loading samples...</div>`;

  try {
    const samples = await API.getDatasetSamples(datasetId, suspiciousOnly, 40, 0);
    if (!samples || samples.length === 0) {
      grid.innerHTML = `<div style="grid-column: 1 / -1; text-align: center; color: var(--text-dim); padding: 2rem;">No ${suspiciousOnly ? 'suspicious ' : ''}samples found for this dataset.</div>`;
      return;
    }

    grid.innerHTML = samples.map(s => `
      <div class="sample-card ${s.is_suspicious ? 'suspicious' : ''}">
        <div class="sample-img-container">
          <img src="${s.image_url}" alt="${s.sample_id}" onerror="this.onerror=null; this.parentElement.innerHTML='<span style=\'color: var(--text-dim); font-size: 0.75rem;\'>Image Missing</span>';">
        </div>
        <div class="sample-meta">
          <div class="id">${s.sample_id}</div>
          <div style="color: var(--text-dim);">${s.source_id} | ${s.labels.join(', ') || 'No label'}</div>
          ${s.is_suspicious 
            ? `<span class="badge badge-critical">SUSPICIOUS</span><div style="font-size: 0.68rem; color: var(--risk-critical); margin-top: 0.2rem;">${s.anomaly_reasons.join(', ')}</div>` 
            : `<span class="badge badge-low">CLEAN</span>`}
        </div>
      </div>
    `).join('');
  } catch (err) {
    grid.innerHTML = `<div style="grid-column: 1 / -1; text-align: center; color: var(--risk-critical); padding: 2rem;">Error loading samples: ${err.message}</div>`;
  }
}
