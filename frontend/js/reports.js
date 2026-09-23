/**
 * VisionTrust Assurance - Reports Page Logic
 */
import { API, showToast, getRiskBadge } from './api.js';

document.addEventListener('DOMContentLoaded', async () => {
  await loadScopes();
  await loadReports();

  const form = document.getElementById('generate-report-form');
  if (form) {
    form.addEventListener('submit', async (e) => {
      e.preventDefault();
      const dsId = document.getElementById('select-report-dataset').value || null;
      const modId = document.getElementById('select-report-model').value || null;
      const title = document.getElementById('report-title-input').value;

      const btn = document.getElementById('btn-compile-report');
      btn.disabled = true;
      btn.innerHTML = '<span class="spinner"></span> Compiling...';

      try {
        const res = await API.generateReport(dsId, modId, title);
        showToast(`Assurance report ${res.report_id} generated successfully!`, 'success');
        await loadReports();
      } catch (err) {
        showToast(`Report generation error: ${err.message}`, 'error');
      } finally {
        btn.disabled = false;
        btn.textContent = 'Compile Dossier';
      }
    });
  }
});

async function loadScopes() {
  try {
    const datasets = await API.listDatasets();
    const dsSelect = document.getElementById('select-report-dataset');
    if (dsSelect) {
      dsSelect.innerHTML = '<option value="">All Registered Datasets</option>' +
        datasets.map(d => `<option value="${d.id}">${d.name} (${d.id})</option>`).join('');
    }

    const models = await API.listModels();
    const modSelect = document.getElementById('select-report-model');
    if (modSelect) {
      modSelect.innerHTML = '<option value="">All Registered Models</option>' +
        models.map(m => `<option value="${m.id}">${m.name} (${m.id})</option>`).join('');
    }
  } catch (err) {
    console.error(err);
  }
}

async function loadReports() {
  try {
    const reports = await API.listReports();
    const tbody = document.getElementById('reports-tbody');
    if (!tbody) return;

    if (!reports || reports.length === 0) {
      tbody.innerHTML = `<tr><td colspan="6" style="text-align: center; color: var(--text-dim); padding: 2rem;">No assurance dossiers compiled yet.</td></tr>`;
      return;
    }

    tbody.innerHTML = reports.map(r => `
      <tr>
        <td><code>${r.report_id}</code></td>
        <td><strong>${r.title}</strong></td>
        <td><strong>${r.overall_risk_score.toFixed(1)}</strong></td>
        <td>${getRiskBadge(r.overall_disposition)}</td>
        <td style="font-size: 0.78rem; color: var(--text-muted);">${new Date(r.created_at).toLocaleString()}</td>
        <td>
          <div style="display: flex; gap: 0.4rem;">
            <a href="${r.pdf_url}" class="btn btn-primary btn-sm" target="_blank">PDF</a>
            <a href="${r.json_url}" class="btn btn-secondary btn-sm" target="_blank">JSON</a>
            <a href="${r.csv_url}" class="btn btn-secondary btn-sm" target="_blank">CSV</a>
          </div>
        </td>
      </tr>
    `).join('');

  } catch (err) {
    showToast(`Failed to load reports: ${err.message}`, 'error');
  }
}
