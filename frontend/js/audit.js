/**
 * VisionTrust Assurance - Audit Trail Page Logic
 */
import { API, showToast } from './api.js';

document.addEventListener('DOMContentLoaded', async () => {
  await loadAuditEvents();
  await runChainVerification();

  const btnVerify = document.getElementById('btn-verify-chain');
  if (btnVerify) {
    btnVerify.addEventListener('click', async () => {
      btnVerify.disabled = true;
      btnVerify.innerHTML = '<span class="spinner"></span> Verifying...';
      try {
        await runChainVerification();
        showToast('Cryptographic audit chain verification completed!', 'success');
      } catch (err) {
        showToast(`Verification failed: ${err.message}`, 'error');
      } finally {
        btnVerify.disabled = false;
        btnVerify.innerHTML = '<span>⛓</span> Verify Full Hash Chain Integrity';
      }
    });
  }
});

async function runChainVerification() {
  try {
    const res = await API.verifyAuditChain();
    const statusText = document.getElementById('chain-status-text');
    const badgeContainer = document.getElementById('chain-badge-container');
    const detailsBox = document.getElementById('chain-details-box');

    if (res.status === 'VALID') {
      statusText.textContent = 'SECURE & IMMUTABLE';
      statusText.style.color = 'var(--risk-low)';
      badgeContainer.innerHTML = '<span class="badge badge-low">CHAIN INTACT</span>';
      detailsBox.textContent = `All ${res.total_events} sequential event hashes verified successfully. Genesis hash invariant ("0"*64) is intact. No insertions, deletions, or bitflips detected.`;
    } else {
      statusText.textContent = 'CORRUPTED / TAMPERED';
      statusText.style.color = 'var(--risk-critical)';
      badgeContainer.innerHTML = '<span class="badge badge-critical">TAMPER DETECTED</span>';
      detailsBox.textContent = `Integrity violation at block ${res.broken_at_index || 'unknown'}: Expected hash does not match computed SHA-256 link.`;
    }
  } catch (err) {
    console.error(err);
  }
}

async function loadAuditEvents() {
  try {
    const events = await API.listAuditEvents(100);
    const tbody = document.getElementById('audit-tbody');
    const countLabel = document.getElementById('events-count-label');
    if (countLabel) countLabel.textContent = `${events.length} events logged`;
    if (!tbody) return;

    if (!events || events.length === 0) {
      tbody.innerHTML = `<tr><td colspan="8" style="text-align: center; color: var(--text-dim); padding: 2rem;">No audit events recorded.</td></tr>`;
      return;
    }

    tbody.innerHTML = events.map(e => `
      <tr>
        <td><code>${e.event_id}</code></td>
        <td style="font-size: 0.78rem; color: var(--text-muted);">${new Date(e.timestamp).toLocaleString()}</td>
        <td><span class="badge badge-info">${e.actor}</span></td>
        <td><strong>${e.action}</strong></td>
        <td><span style="font-size: 0.8rem; color: var(--text-dim);">${e.asset}</span></td>
        <td><code style="font-size: 0.7rem; color: var(--text-dim);">${e.previous_hash.substring(0, 12)}...</code></td>
        <td><code style="font-size: 0.7rem; color: var(--accent-cyan);">${e.current_hash.substring(0, 12)}...</code></td>
        <td>
          <button class="btn btn-secondary btn-sm btn-meta-view" data-meta='${JSON.stringify(e.metadata)}'>
            Meta
          </button>
        </td>
      </tr>
    `).join('');

    document.querySelectorAll('.btn-meta-view').forEach(btn => {
      btn.addEventListener('click', (ev) => {
        const metaStr = ev.target.getAttribute('data-meta');
        alert(`Audit Metadata:\n\n${JSON.stringify(JSON.parse(metaStr), null, 2)}`);
      });
    });

  } catch (err) {
    showToast(`Failed to load audit events: ${err.message}`, 'error');
  }
}
