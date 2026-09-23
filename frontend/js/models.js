/**
 * VisionTrust Assurance - Models Page Logic
 * Handles Model Upload, Fingerprinting, Substitution Detection & Behavioral Perturbation Battery
 */
import { API, showToast, getRiskBadge } from './api.js';

let registeredModels = [];

document.addEventListener('DOMContentLoaded', async () => {
  await loadModels();

  // Setup file dropzone
  const fileInput = document.getElementById('model-file-input');
  const filenameLabel = document.getElementById('dropzone-model-name');
  if (fileInput && filenameLabel) {
    fileInput.addEventListener('change', () => {
      if (fileInput.files.length > 0) {
        filenameLabel.textContent = fileInput.files[0].name;
      }
    });
  }

  // Upload Form
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
        btn.innerHTML = '<span>◆</span> Inspect & Register Model';
      }
    });
  }

  // Behavioral Battery Form
  const batteryForm = document.getElementById('battery-form');
  if (batteryForm) {
    batteryForm.addEventListener('submit', async (e) => {
      e.preventDefault();
      const refId = document.getElementById('select-ref-model').value;
      const candId = document.getElementById('select-cand-model').value;
      const samples = parseInt(document.getElementById('input-battery-samples').value) || 24;

      if (!candId) {
        showToast('Please select a candidate model to evaluate', 'error');
        return;
      }

      const btn = document.getElementById('btn-run-battery');
      btn.disabled = true;
      btn.innerHTML = '<span class="spinner"></span> Evaluating Battery...';

      try {
        showToast(`Running behavioral battery on ${candId} (tests: ${samples})...`, 'info');
        const res = await API.analyzeModel(candId, refId || null, samples);
        showToast(`Behavioral battery complete! Risk: ${res.overall_risk_score.toFixed(1)}`, 'success');
        displayModelAnalysisResults(res);
        await loadModels();
      } catch (err) {
        showToast(`Battery execution error: ${err.message}`, 'error');
      } finally {
        btn.disabled = false;
        btn.innerHTML = '<span>▶</span> Run Behavioral Battery';
      }
    });
  }
});

async function loadModels() {
  try {
    const models = await API.listModels();
    registeredModels = models || [];
    const tbody = document.getElementById('models-tbody');
    const selectRef = document.getElementById('select-ref-model');
    const selectCand = document.getElementById('select-cand-model');

    // Populate dropdowns
    if (selectRef && selectCand) {
      const prevRef = selectRef.value;
      const prevCand = selectCand.value;

      const optionsHtml = '<option value="">Select Model...</option>' + 
        registeredModels.map(m => `<option value="${m.id}">${m.name} (${m.framework})</option>`).join('');

      selectRef.innerHTML = optionsHtml;
      selectCand.innerHTML = optionsHtml;

      // Auto-select smart defaults if available
      const refDefault = registeredModels.find(m => m.name.toLowerCase().includes('reference'));
      const candDefault = registeredModels.find(m => m.name.toLowerCase().includes('substituted') || m.name.toLowerCase().includes('candidate'));

      if (refDefault) selectRef.value = prevRef || refDefault.id;
      if (candDefault) selectCand.value = prevCand || candDefault.id;
    }

    if (!tbody) return;

    if (!registeredModels || registeredModels.length === 0) {
      tbody.innerHTML = `<tr><td colspan="10" style="text-align: center; color: var(--text-dim); padding: 2rem;">No models registered. Upload a model above.</td></tr>`;
      return;
    }

    tbody.innerHTML = registeredModels.map(m => `
      <tr>
        <td><code>${m.id}</code></td>
        <td><strong>${m.name}</strong></td>
        <td><span class="badge badge-info">${m.framework}</span></td>
        <td><code style="font-size: 0.72rem; color: var(--color-primary);">${m.sha256 ? m.sha256.substring(0, 16) + '...' : 'N/A'}</code></td>
        <td>${m.parameter_count ? m.parameter_count.toLocaleString() : 'N/A'}</td>
        <td><span class="badge ${m.access_level === 'WHITE_BOX' ? 'badge-low' : 'badge-med'}">${m.access_level}</span></td>
        <td><span class="badge ${m.status === 'ANALYZED' ? 'badge-low' : 'badge-med'}">${m.status}</span></td>
        <td><strong>${m.risk_score ? m.risk_score.toFixed(1) : '0.0'}</strong></td>
        <td>${getRiskBadge(m.disposition)}</td>
        <td>
          <div style="display: flex; gap: 0.5rem;">
            <button class="btn btn-secondary btn-sm btn-inspect-fp" data-id="${m.id}" title="Inspect Cryptographic Fingerprint">
              Fingerprint
            </button>
            <button class="btn btn-primary btn-sm btn-analyze-model" data-id="${m.id}" title="Run Behavioral Battery">
              Run Battery
            </button>
          </div>
        </td>
      </tr>
    `).join('');

    // Attach listeners
    document.querySelectorAll('.btn-inspect-fp').forEach(btn => {
      btn.addEventListener('click', async (e) => {
        const id = e.target.getAttribute('data-id');
        await showModelFingerprint(id);
      });
    });

    document.querySelectorAll('.btn-analyze-model').forEach(btn => {
      btn.addEventListener('click', async (e) => {
        const id = e.target.getAttribute('data-id');
        const refModel = registeredModels.find(m => m.id !== id && m.name.toLowerCase().includes('reference'));
        const refId = refModel ? refModel.id : null;
        await triggerModelAnalysis(id, refId, e.target);
      });
    });

  } catch (err) {
    showToast(`Failed to load models: ${err.message}`, 'error');
  }
}

async function showModelFingerprint(modelId) {
  try {
    showToast(`Loading fingerprint for ${modelId}...`, 'info');
    const res = await API.getModel(modelId);
    const panel = document.getElementById('model-analysis-panel');
    if (!panel) return;

    panel.style.display = 'block';
    document.getElementById('res-model-id').textContent = `${res.name} (${res.id})`;
    document.getElementById('res-model-disposition').innerHTML = getRiskBadge(res.disposition);
    document.getElementById('res-sub-status').textContent = 'FINGERPRINTED';
    document.getElementById('res-sub-status').style.color = 'var(--color-primary)';
    document.getElementById('res-sub-detail').textContent = `SHA-256: ${res.sha256}`;
    document.getElementById('res-battery-agreement').textContent = `${(res.parameter_count || 0).toLocaleString()} params`;
    document.getElementById('res-battery-tests').textContent = `Access: ${res.access_level}`;
    document.getElementById('res-trigger-sens').textContent = res.framework;
    document.getElementById('res-trigger-conf').textContent = `Input: ${res.input_shape}`;

    const fpBlock = document.getElementById('model-fp-evidence');
    if (fpBlock) {
      fpBlock.textContent = JSON.stringify(
        {
          id: res.id,
          name: res.name,
          framework: res.framework,
          sha256: res.sha256,
          file_size_bytes: res.file_size_bytes,
          parameter_count: res.parameter_count,
          access_level: res.access_level,
          input_shape: res.input_shape,
          output_shape: res.output_shape,
          architecture: res.architecture,
          fingerprint: res.fingerprint
        },
        null,
        2
      );
    }
    panel.scrollIntoView({ behavior: 'smooth' });
  } catch (err) {
    showToast(`Fingerprint lookup failed: ${err.message}`, 'error');
  }
}

async function triggerModelAnalysis(modelId, refId, btnEl) {
  btnEl.disabled = true;
  btnEl.innerHTML = '<span class="spinner"></span> Analyzing...';

  try {
    showToast(`Evaluating model ${modelId} against baseline ${refId || 'N/A'}...`, 'info');
    const res = await API.analyzeModel(modelId, refId, 24);
    showToast(`Model analysis complete for ${modelId}! Score: ${res.overall_risk_score.toFixed(1)}`, 'success');
    displayModelAnalysisResults(res);
    await loadModels();
  } catch (err) {
    showToast(`Model analysis error: ${err.message}`, 'error');
  } finally {
    btnEl.disabled = false;
    btnEl.textContent = 'Run Battery';
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
