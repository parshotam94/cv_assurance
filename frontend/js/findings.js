/**
 * VisionTrust Assurance - Findings Page Logic
 */
import { API, showToast, getRiskBadge } from './api.js';

let allFindings = [];

document.addEventListener('DOMContentLoaded', async () => {
  await loadFindings();

  // Filter listeners
  const filterSev = document.getElementById('filter-severity');
  const filterCat = document.getElementById('filter-category');
  const filterAsset = document.getElementById('filter-asset');

  if (filterSev) filterSev.addEventListener('change', () => applyFilters());
  if (filterCat) filterCat.addEventListener('change', () => applyFilters());
  if (filterAsset) filterAsset.addEventListener('input', () => applyFilters());

  // Drawer close listeners
  const btnClose = document.getElementById('btn-close-drawer');
  const overlay = document.getElementById('drawer-overlay');
  if (btnClose) btnClose.addEventListener('click', () => closeDrawer());
  if (overlay) overlay.addEventListener('click', () => closeDrawer());
});

async function loadFindings() {
  try {
    const findings = await API.listFindings();
    allFindings = findings;
    applyFilters();
  } catch (err) {
    showToast(`Failed to load findings: ${err.message}`, 'error');
  }
}

function applyFilters() {
  const sev = document.getElementById('filter-severity').value;
  const cat = document.getElementById('filter-category').value;
  const asset = document.getElementById('filter-asset').value.trim().toLowerCase();

  let filtered = allFindings;

  if (sev) {
    filtered = filtered.filter(f => f.severity === sev);
  }
  if (cat) {
    filtered = filtered.filter(f => f.category === cat);
  }
  if (asset) {
    filtered = filtered.filter(f => (f.asset_id || '').toLowerCase().includes(asset));
  }

  renderFindingsTable(filtered);
}

function renderFindingsTable(findings) {
  const tbody = document.getElementById('findings-tbody');
  if (!tbody) return;

  if (!findings || findings.length === 0) {
    tbody.innerHTML = `<tr><td colspan="9" style="text-align: center; color: var(--text-dim); padding: 2rem;">No findings matching current filters.</td></tr>`;
    return;
  }

  tbody.innerHTML = findings.map(f => `
    <tr>
      <td><code style="color: var(--accent-cyan);">${f.finding_id}</code></td>
      <td><span style="font-size: 0.78rem; color: var(--text-dim);">${f.asset_type}:</span> ${f.asset_id}</td>
      <td><span class="badge badge-info">${f.category}</span></td>
      <td>${getRiskBadge(f.severity)}</td>
      <td>${(f.confidence * 100).toFixed(0)}%</td>
      <td><strong>${f.risk_score.toFixed(1)}</strong></td>
      <td>${f.title}</td>
      <td>${getRiskBadge(f.recommendation)}</td>
      <td>
        <button class="btn btn-secondary btn-sm btn-open-finding" data-id="${f.finding_id}">
          Evidence
        </button>
      </td>
    </tr>
  `).join('');

  document.querySelectorAll('.btn-open-finding').forEach(btn => {
    btn.addEventListener('click', async (e) => {
      const id = e.target.getAttribute('data-id');
      await openFindingDrawer(id);
    });
  });
}

async function openFindingDrawer(findingId) {
  try {
    const f = await API.getFinding(findingId);
    
    document.getElementById('drawer-title').textContent = f.title;
    document.getElementById('drawer-subtitle').textContent = `${f.finding_id} | ${f.asset_type}: ${f.asset_id} | ${f.category}`;
    document.getElementById('drawer-desc').textContent = f.description;
    document.getElementById('drawer-method').textContent = f.detection_method;

    // Evidence
    const evBlock = document.getElementById('drawer-evidence');
    if (evBlock) {
      evBlock.textContent = f.evidence && f.evidence.length > 0 ? f.evidence.join('\n\n') : 'No additional telemetry recorded.';
    }

    // Samples
    const samplesBlock = document.getElementById('drawer-samples');
    if (samplesBlock) {
      samplesBlock.textContent = f.affected_samples && f.affected_samples.length > 0 ? f.affected_samples.join(', ') : 'Asset-level scope (no isolated sample list).';
    }

    // Limitations
    const limBlock = document.getElementById('drawer-limitations');
    if (limBlock) {
      limBlock.textContent = f.limitations && f.limitations.length > 0 ? f.limitations.join('\n') : 'Standard algorithmic constraints apply.';
    }

    // Recommendation
    document.getElementById('drawer-rec').innerHTML = `${getRiskBadge(f.recommendation)} &nbsp; Risk Score: ${f.risk_score.toFixed(1)}`;

    document.getElementById('evidence-drawer').classList.add('open');
    document.getElementById('drawer-overlay').classList.add('open');
  } catch (err) {
    showToast(`Error retrieving finding details: ${err.message}`, 'error');
  }
}

function closeDrawer() {
  document.getElementById('evidence-drawer').classList.remove('open');
  document.getElementById('drawer-overlay').classList.remove('open');
}
