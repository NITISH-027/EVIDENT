/**
 * EVIDENT - Review Verdict and Engineer Notes Flow
 * Progressive enhancement: works with standard form submit or async fetch.
 */
(function() {
  function initVerdictForm() {
    const form = document.getElementById('engineer-verdict-form');
    if (!form) return;

    const liveRegion = document.getElementById('verdict-confirmation-live');

    form.addEventListener('submit', async (e) => {
      e.preventDefault();
      const submitter = e.submitter;
      if (!submitter || !submitter.value) return;

      const verdictVal = submitter.value;
      const formData = new FormData(form);
      formData.set('verdict', verdictVal);

      const subset = formData.get('subset');
      const unit = formData.get('unit');
      const cycle = formData.get('cycle');
      const policy = formData.get('policy');
      const version = formData.get('version');

      // Disable buttons during submission
      const buttons = form.querySelectorAll('button[name="verdict"]');
      buttons.forEach(b => b.disabled = true);

      try {
        const resp = await fetch('/verify', {
          method: 'POST',
          body: formData
        });

        const data = await resp.json();
        if (!resp.ok) {
          throw new Error(data.error || `Server returned ${resp.status}`);
        }

        // Success
        const entry = data.logged;
        const vLabel = entry.verdict === 'reviewed_warrants_inspection' ? 'Warrants Inspection' : 'Dismissed';
        
        if (liveRegion) {
          liveRegion.className = 'confirmation-live visible';
          liveRegion.innerHTML = `
            <b>Verdict recorded:</b> ${vLabel} (engineer opinion) for ${entry.subset} unit ${entry.unit} at cycle ${entry.cycle} [policy: ${entry.policy}] at <span style="font-family:var(--font-mono);">${entry.ts}</span>.
          `;
        }

        // Update queue row
        if (window.evidentQueue) {
          window.evidentQueue.markUnitReviewed(entry.unit, entry.cycle, entry.verdict, entry.policy);
        }

        // Also add to past verdicts list in evidence panel
        const pastVerdictsEl = document.getElementById('past-verdicts-list');
        if (pastVerdictsEl) {
          const newLi = document.createElement('li');
          newLi.style = 'font-size:0.75rem; color:var(--text-muted); border-bottom:1px solid var(--border-muted); padding-bottom:2px;';
          newLi.innerHTML = `<b>${vLabel}</b> (engineer opinion) at cycle ${entry.cycle} [${entry.policy}] · <span style="font-family:var(--font-mono);">${entry.ts}</span>`;
          const ul = pastVerdictsEl.querySelector('ul');
          if (ul) {
            ul.prepend(newLi);
          } else {
            pastVerdictsEl.innerHTML = `<ul style="list-style:none; padding:0; display:flex; flex-direction:column; gap:0.25rem;"></ul>`;
            pastVerdictsEl.querySelector('ul').appendChild(newLi);
          }
        }
      } catch (err) {
        if (liveRegion) {
          liveRegion.className = 'caveat-box';
          liveRegion.style.display = 'block';
          liveRegion.textContent = `Error recording verdict: ${err.message}`;
        }
      } finally {
        buttons.forEach(b => b.disabled = false);
      }
    });
  }

  // Engineer Notes Management
  function initNotes() {
    const noteForm = document.getElementById('engineer-note-form');
    const notesList = document.getElementById('engineer-notes-list');
    if (!noteForm || !notesList) return;

    noteForm.addEventListener('submit', async (e) => {
      e.preventDefault();
      const input = document.getElementById('note-text-input');
      const text = input ? input.value.trim() : '';
      if (!text) return;

      const subset = document.getElementById('verdict-subset')?.value || 'FD001';
      const unit = document.getElementById('verdict-unit')?.value;
      const cycle = document.getElementById('verdict-cycle')?.value;
      const policy = document.getElementById('verdict-policy')?.value || 'v1_q50';

      try {
        const resp = await fetch('/api/note', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ subset, unit: parseInt(unit, 10), cycle: parseInt(cycle, 10), policy, text })
        });
        const res = await resp.json();
        if (resp.ok && res.logged) {
          input.value = '';
          loadNotes(subset, unit);
        }
      } catch (err) {
        console.warn('Note save failed', err);
      }
    });

    window.loadNotes = async function(subset, unit) {
      if (!notesList || !subset || !unit) return;
      try {
        const resp = await fetch(`/api/notes?subset=${subset}&unit=${unit}`);
        if (!resp.ok) return;
        const data = await resp.json();
        if (data.notes && data.notes.length) {
          notesList.innerHTML = data.notes.map(n => `
            <div class="note-item">
              <div class="note-meta">${n.ts} · Cycle ${n.cycle} · ${n.label}</div>
              <div>${escapeHtml(n.text)}</div>
            </div>
          `).join('');
        } else {
          notesList.innerHTML = '<span style="font-size:0.75rem; color:var(--text-dim);">No engineer notes recorded for this unit.</span>';
        }
      } catch (e) {}
    };
  }

  function escapeHtml(str) {
    const div = document.createElement('div');
    div.textContent = str;
    return div.innerHTML;
  }

  document.addEventListener('DOMContentLoaded', () => {
    initVerdictForm();
    initNotes();
  });
})();

