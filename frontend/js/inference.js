/**
 * VisionTrust Assurance - Inference & Provenance Page Logic
 */
import { API, showToast, getRiskBadge } from './api.js';

let activeRecord = null;

document.addEventListener('DOMContentLoaded', async () => {
  await loadModelOptions();
  await loadInferenceRecords();

  const form = document.getElementById('run-inference-form');
  if (form) {
    form.addEventListener('submit', async (e) => {
      e.preventDefault();
      const modelId = document.getElementById('select-inference-model').value;
      const fileInput = document.getElementById('inference-image-input');
      const file = fileInput.files[0];

      if (!modelId || !file) {
        showToast('Please select a model and an image', 'error');
        return;
      }

      const btn = document.getElementById('btn-run-inference');
      btn.disabled = true;
      btn.innerHTML = '<span class="spinner"></span> Running & Signing...';

      try {
        const res = await API.runInference(file, modelId);
        showToast(`Inference executed! Record ID: ${res.record.record_id}`, 'success');
        activeRecord = res.record;
        displayActiveRecord(res.record, 'VERIFIED');
        await loadInferenceRecords();
      } catch (err) {
        showToast(`Inference error: ${err.message}`, 'error');
      } finally {
        btn.disabled = false;
        btn.textContent = 'Run Inference & Generate Provenance Record';
      }
    });
  }

  // Verify Active Record
  const btnVerify = document.getElementById('btn-verify-active');
  if (btnVerify) {
    btnVerify.addEventListener('click', async () => {
      if (!activeRecord) return;
      btnVerify.disabled = true;
      try {
        const verif = await API.verifyInference(activeRecord);
        showToast(`Verification: ${verif.status} (Valid: ${verif.is_valid})`, verif.is_valid ? 'success' : 'error');
        displayActiveRecord(activeRecord, verif.status);
        await loadInferenceRecords();
      } catch (err) {
        showToast(`Verification error: ${err.message}`, 'error');
      } finally {
        btnVerify.disabled = false;
      }
    });
  }

  // Tamper Test
  const btnTamper = document.getElementById('btn-tamper-test');
  if (btnTamper) {
    btnTamper.addEventListener('click', async () => {
      if (!activeRecord) {
        showToast('Please run or select an inference record first', 'error');
        return;
      }
      const field = document.getElementById('select-tamper-field').value;
      btnTamper.disabled = true;
      try {
        const res = await API.simulateTampering(activeRecord.record_id, field);
        showToast(`Tampering injected in '${field}'! Verification status: ${res.verification_result.status}`, 'error');
        displayActiveRecord(activeRecord, res.verification_result.status);
        await loadInferenceRecords();
      } catch (err) {
        showToast(`Tamper simulation error: ${err.message}`, 'error');
      } finally {
        btnTamper.disabled = false;
      }
    });
  }

  // Replay Test
  const btnReplay = document.getElementById('btn-replay-test');
  if (btnReplay) {
    btnReplay.addEventListener('click', async () => {
      if (!activeRecord) {
        showToast('Please run or select an inference record first', 'error');
        return;
      }
      btnReplay.disabled = true;
      try {
        const res = await API.simulateReplay(activeRecord.record_id);
        if (res.is_replay_detected) {
          showToast(`Replay attack detected! Reasons: ${res.reasons.join(', ')}`, 'error');
          displayActiveRecord(activeRecord, 'REPLAY_DETECTED');
        } else {
          showToast('Replay was unexpectedly accepted', 'error');
        }
        await loadInferenceRecords();
      } catch (err) {
        showToast(`Replay simulation error: ${err.message}`, 'error');
      } finally {
        btnReplay.disabled = false;
      }
    });
  }
});

async function loadModelOptions() {
  try {
    const models = await API.listModels();
    const select = document.getElementById('select-inference-model');
    if (select) {
      select.innerHTML = '<option value="">Select registered model...</option>' +
        models.map(m => `<option value="${m.id}">${m.name} (${m.framework})</option>`).join('');
    }
  } catch (err) {
    console.error(err);
  }
}

async function loadInferenceRecords() {
  try {
    const records = await API.listInferenceRecords(50);
    const tbody = document.getElementById('inference-records-tbody');
    if (!tbody) return;

    if (!records || records.length === 0) {
      tbody.innerHTML = `<tr><td colspan="9" style="text-align: center; color: var(--text-dim); padding: 2rem;">No inference records stored. Execute an inference above.</td></tr>`;
      return;
    }

    tbody.innerHTML = records.map(r => `
      <tr>
        <td><code>${r.record_id}</code></td>
        <td><strong>#${r.sequence}</strong></td>
        <td><code style="font-size: 0.72rem; color: var(--text-dim);">${r.model_sha256 ? r.model_sha256.substring(0, 12) + '...' : ''}</code></td>
        <td><code style="font-size: 0.72rem; color: var(--text-dim);">${r.input_sha256 ? r.input_sha256.substring(0, 12) + '...' : ''}</code></td>
        <td style="font-size: 0.78rem; color: var(--text-muted);">${r.timestamp ? r.timestamp.substring(11, 19) : ''}</td>
        <td><span class="badge ${r.verification_status === 'VERIFIED' ? 'badge-low' : 'badge-critical'}">${r.verification_status}</span></td>
        <td>${r.is_tampered ? '<span class="badge badge-critical">YES</span>' : '<span style="color: var(--text-dim);">No</span>'}</td>
        <td>${r.is_replayed ? '<span class="badge badge-critical">YES</span>' : '<span style="color: var(--text-dim);">No</span>'}</td>
        <td>
          <button class="btn btn-secondary btn-sm btn-select-record" data-id="${r.record_id}">
            Select
          </button>
        </td>
      </tr>
    `).join('');

    document.querySelectorAll('.btn-select-record').forEach(btn => {
      btn.addEventListener('click', (e) => {
        const id = e.target.getAttribute('data-id');
        const found = records.find(x => x.record_id === id);
        if (found) {
          activeRecord = found;
          displayActiveRecord(found, found.verification_status);
        }
      });
    });

  } catch (err) {
    showToast(`Failed to load inference records: ${err.message}`, 'error');
  }
}

function displayActiveRecord(record, status) {
  const panel = document.getElementById('provenance-active-panel');
  if (!panel) return;

  panel.style.display = 'block';
  document.getElementById('active-rec-id').textContent = record.record_id;
  
  const badge = document.getElementById('active-rec-badge');
  badge.className = `badge ${status === 'VERIFIED' ? 'badge-low' : 'badge-critical'}`;
  badge.textContent = status;

  const jsonBlock = document.getElementById('active-record-json');
  jsonBlock.textContent = JSON.stringify(record, null, 2);

  panel.scrollIntoView({ behavior: 'smooth' });
}
