/**
 * VisionTrust Assurance - Dashboard Logic
 */
import { API, showToast, getRiskBadge } from './api.js';

document.addEventListener('DOMContentLoaded', async () => {
  await loadDashboardData();

  const btnDemo = document.getElementById('btn-run-demo');
  if (btnDemo) {
    btnDemo.addEventListener('click', async () => {
      btnDemo.disabled = true;
      btnDemo.innerHTML = '<span class="spinner"></span> Running Benchmark...';
      try {
        showToast('Initiating 8-Attack Benchmark test suite...', 'info');
        const res = await API.runDemoBenchmark();
        showToast(`Benchmark completed: ${res.detected_scenarios}/${res.total_scenarios} attacks flagged!`, 'success');
        await loadDashboardData();
      } catch (err) {
        showToast(`Benchmark run error: ${err.message}`, 'error');
      } finally {
        btnDemo.disabled = false;
        btnDemo.innerHTML = '<span>▶</span> Run Full 8-Attack Benchmark';
      }
    });
  }
});

async function loadDashboardData() {
  // 1. Health
  try {
    const health = await API.getHealth();
    document.getElementById('health-indicator').className = 'badge badge-low';
    document.getElementById('health-indicator').textContent = 'AIR-GAPPED';
    document.getElementById('runtime-status').textContent = 'OPERATIONAL';
    document.getElementById('runtime-info').textContent = `PyTorch ${health.ml_runtimes.pytorch} | ONNX ${health.ml_runtimes.onnxruntime}`;
  } catch (err) {
    document.getElementById('health-indicator').className = 'badge badge-critical';
    document.getElementById('health-indicator').textContent = 'OFFLINE';
  }

  // 2. Datasets
  try {
    const datasets = await API.listDatasets();
    document.getElementById('stat-datasets-count').textContent = datasets.length;
    const analyzed = datasets.filter(d => d.status === 'ANALYZED').length;
    document.getElementById('stat-datasets-sub').textContent = `${analyzed} analyzed (${datasets.length - analyzed} pending)`;
  } catch (err) {
    console.error(err);
  }

  // 3. Models
  try {
    const models = await API.listModels();
    document.getElementById('stat-models-count').textContent = models.length;
    const quarantined = models.filter(m => m.disposition === 'QUARANTINE').length;
    document.getElementById('stat-models-sub').textContent = quarantined > 0 ? `${quarantined} QUARANTINED` : 'All Accepted';
  } catch (err) {
    console.error(err);
  }

  // 4. Inference Records
  try {
    const records = await API.listInferenceRecords(100);
    document.getElementById('stat-inference-count').textContent = records.length;
    const tampered = records.filter(r => r.is_tampered).length;
    const replayed = records.filter(r => r.is_replayed).length;
    document.getElementById('stat-inference-sub').textContent = `${tampered} Tampered | ${replayed} Replayed`;
  } catch (err) {
    console.error(err);
  }

  // 5. Findings Table
  try {
    const findings = await API.listFindings();
    const tbody = document.getElementById('dashboard-findings-tbody');
    if (!tbody) return;

    if (!findings || findings.length === 0) {
      tbody.innerHTML = `<tr><td colspan="7" style="text-align: center; color: var(--text-dim); padding: 2rem;">No active findings recorded. System is secure.</td></tr>`;
    } else {
      // Sort by risk_score desc
      const sorted = [...findings].sort((a, b) => b.risk_score - a.risk_score).slice(0, 5);
      tbody.innerHTML = sorted.map(f => `
        <tr>
          <td><code style="color: var(--accent-cyan);">${f.finding_id}</code></td>
          <td><span style="font-size: 0.8rem; color: var(--text-dim);">${f.asset_type}:</span> ${f.asset_id}</td>
          <td><span class="badge badge-info">${f.category}</span></td>
          <td>${getRiskBadge(f.severity)}</td>
          <td><strong>${f.risk_score.toFixed(1)}</strong></td>
          <td>${f.title}</td>
          <td>${getRiskBadge(f.recommendation)}</td>
        </tr>
      `).join('');
    }
  } catch (err) {
    console.error(err);
  }

  // 6. Audit Status
  try {
    const auditRes = await API.verifyAuditChain();
    const chainBadge = document.getElementById('audit-chain-badge');
    if (chainBadge) {
      if (auditRes.status === 'VALID') {
        chainBadge.className = 'badge badge-low';
        chainBadge.textContent = 'SECURE (VALID)';
      } else {
        chainBadge.className = 'badge badge-critical';
        chainBadge.textContent = 'TAMPERED';
      }
    }
    const events = await API.listAuditEvents(5);
    document.getElementById('audit-events-count').textContent = auditRes.total_events || events.length;
    if (events.length > 0) {
      document.getElementById('audit-root-hash').textContent = events[0].current_hash.substring(0, 24) + '...';
    }
  } catch (err) {
    console.error(err);
  }

  // 7. Latest Report
  try {
    const reports = await API.listReports();
    const reportCard = document.getElementById('latest-report-card');
    if (reportCard && reports.length > 0) {
      const latest = reports[0];
      reportCard.innerHTML = `
        <div style="margin-bottom: 0.75rem;">
          <div style="font-weight: 700; color: #fff;">${latest.title}</div>
          <div style="font-size: 0.75rem; color: var(--text-dim);">${latest.report_id} - ${new Date(latest.created_at).toLocaleString()}</div>
        </div>
        <div style="display: flex; gap: 1rem; align-items: center; margin-bottom: 1rem;">
          <div>Risk Score: <strong>${latest.overall_risk_score.toFixed(1)}</strong></div>
          <div>Disposition: ${getRiskBadge(latest.overall_disposition)}</div>
        </div>
        <div style="display: flex; gap: 0.5rem;">
          <a href="${latest.pdf_url}" class="btn btn-secondary btn-sm" target="_blank">Download PDF</a>
          <a href="${latest.json_url}" class="btn btn-secondary btn-sm" target="_blank">JSON</a>
          <a href="${latest.csv_url}" class="btn btn-secondary btn-sm" target="_blank">CSV</a>
        </div>
      `;
    }
  } catch (err) {
    console.error(err);
  }
}
