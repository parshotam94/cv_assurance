/**
 * VisionTrust Assurance - Models Page Logic
 */
import { API, showToast, getRiskBadge } from './api.js';

let registeredModels = [];

document.addEventListener('DOMContentLoaded', async () => {
  await loadModels();

  const fileInput = document.getElementById('model-file-input');
  const filenameLabel = document.getElementById('dropzone-model-name');
  if (fileInput && filenameLabel) {
    fileInput.addEventListener('change', () => {
      if (fileInput.files.length > 0) {
        filenameLabel.textContent = fileInput.files[0].name;
      }
    });
  }

  const form = document.getElementById('upload-model-form');
  if (form) {
    form.addEventListener('submit', async (e) => {
      e.preventDefault();
      const file = fileInput.files[0];
      if (!file) {
        showToast('Please select a model file (.onnx, .pt)', 'error');
        return;
      }
      const name = document.getElementById('model-name-input').value;
      const btn = document.getElementById('btn-upload-model');
      btn.disabled = true;
      btn.innerHTML = '<span class="spinner"></span> Inspecting...';

      try {
        const res = await API.uploadModel(file, name);
        showToast(`Model ${res.model_id} registered! (${res.parameter_count.toLocaleString()} params)`, 'success');
        form.reset();
        filenameLabel.textContent = 'Drag & drop model file or click to select';
        await loadModels();
      } catch (err) {
        showToast(`Upload error: ${err.message}`, 'error');
      } finally {
        btn.disabled = false;
        btn.textContent = 'Inspect & Register Model';
      }
    });
  }
});

async function loadModels() {
  try {
    const models = await API.listModels();
    registeredModels = models;
    const tbody = document.getElementById('models-tbody');
    if (!tbody) return;

    if (!models || models.length === 0) {
      tbody.innerHTML = `<tr><td colspan="10" style="text-align: center; color: var(--text-dim); padding: 2rem;">No models registered. Upload a model above.</td></tr>`;
      return;
    }

    tbody.innerHTML = models.map(m => `
      <tr>
        <td><code>${m.id}</code></td>
        <td><strong>${m.name}</strong></td>
        <td><span class="badge badge-info">${m.framework}</span></td>
        <td><code style="font-size: 0.72rem; color: var(--text-dim);">${m.sha256 ? m.sha256.substring(0, 16) + '...' : 'N/A'}</code></td>
        <td>${m.parameter_count ? m.parameter_count.toLocaleString() : 'N/A'}</td>
        <td><span class="badge ${m.access_level === 'WHITE_BOX' ? 'badge-low' : 'badge-med'}">${m.access_level}</span></td>
        <td><span class="badge ${m.status === 'ANALYZED' ? 'badge-low' : 'badge-med'}">${m.status}</span></td>
        <td><strong>${m.risk_score ? m.risk_score.toFixed(1) : '0.0'}</strong></td>
        <td>${getRiskBadge(m.disposition)}</td>
        <td>
          <button class="btn btn-primary btn-sm btn-analyze-model" data-id="${m.id}">
            Analyze
          </button>
        </td>
      </tr>
    `).join('');

    document.querySelectorAll('.btn-analyze-model').forEach(btn => {
      btn.addEventListener('click', async (e) => {
        const id = e.target.getAttribute('data-id');
        await triggerModelAnalysis(id, e.target);
      });
    });

  } catch (err) {
    showToast(`Failed to load models: ${err.message}`, 'error');
  }
}

async function triggerModelAnalysis(modelId, btnEl) {
  btnEl.disabled = true;
  btnEl.innerHTML = '<span class="spinner"></span> Analyzing...';

  // Automatically find reference model if available
  const refModel = registeredModels.find(m => m.id !== modelId && m.name.toLowerCase().includes('reference'));
  const refId = refModel ? refModel.id : null;

  try {
    showToast(`Evaluating model ${modelId}${refId ? ` against baseline ${refId}` : ''}...`, 'info');
    const res = await API.analyzeModel(modelId, refId, 40);
    showToast(`Model analysis complete for ${modelId}! Score: ${res.overall_risk_score.toFixed(1)}`, 'success');
    
    // Display results
    displayModelAnalysisResults(res);
    await loadModels();
  } catch (err) {
    showToast(`Model analysis error: ${err.message}`, 'error');
  } finally {
    btnEl.disabled = false;
    btnEl.textContent = 'Analyze';
  }
}

function displayModelAnalysisResults(res) {
  const panel = document.getElementById('model-analysis-panel');
  if (!panel) return;

  panel.style.display = 'block';
  document.getElementById('res-model-id').textContent = res.model_id;
  document.getElementById('res-model-disposition').innerHTML = getRiskBadge(res.disposition);

  // Substitution
  const sub = res.substitution_analysis;
  if (sub) {
    document.getElementById('res-sub-status').textContent = sub.status;
    document.getElementById('res-sub-status').style.color = sub.status === 'MATCH' ? 'var(--risk-low)' : 'var(--risk-critical)';
    document.getElementById('res-sub-detail').textContent = sub.details;
  } else {
    document.getElementById('res-sub-status').textContent = 'STANDALONE';
    document.getElementById('res-sub-detail').textContent = 'No baseline model provided for differential comparison';
  }

  // Battery
  const bat = res.behavioural_battery;
  if (bat && bat.behavioural_agreement_rate !== null) {
    document.getElementById('res-battery-agreement').textContent = `${(bat.behavioural_agreement_rate * 100).toFixed(0)}%`;
    document.getElementById('res-battery-tests').textContent = `${bat.anomalous_inputs_count} divergences on ${bat.total_battery_tests} tests`;
  } else {
    document.getElementById('res-battery-agreement').textContent = 'N/A';
  }

  // Trigger
  const trg = res.trigger_sensitivity;
  if (trg) {
    document.getElementById('res-trigger-sens').textContent = trg.high_vulnerability_detected ? 'VULNERABLE' : 'ROBUST';
    document.getElementById('res-trigger-sens').style.color = trg.high_vulnerability_detected ? 'var(--risk-high)' : 'var(--risk-low)';
    document.getElementById('res-trigger-conf').textContent = `Confidence: ${(trg.confidence * 100).toFixed(0)}%`;
  }

  // Fingerprint / Details
  const fpBlock = document.getElementById('model-fp-evidence');
  if (fpBlock) {
    fpBlock.textContent = JSON.stringify(
      {
        access_level: res.access_level,
        substitution_analysis: res.substitution_analysis,
        behavioural_summary: res.behavioural_battery,
        trigger_summary: res.trigger_sensitivity,
        activation_analysis: res.activation_analysis
      },
      null,
      2
    );
  }

  panel.scrollIntoView({ behavior: 'smooth' });
}
