/**
 * EVIDENT - Keyboard Shortcuts Sheet & Global Handler
 */
(function() {
  function createShortcutsModal() {
    const modal = document.createElement('div');
    modal.id = 'shortcuts-modal';
    modal.className = 'drawer-backdrop';
    modal.innerHTML = `
      <div class="card" style="position:fixed; top:50%; left:50%; transform:translate(-50%, -50%); max-width:480px; width:90%; z-index:10001; background:var(--bg-surface-elevated); box-shadow:var(--shadow-md);" role="dialog" aria-label="Keyboard Shortcuts">
        <div class="card-header">
          <div class="card-title">Keyboard Shortcuts</div>
          <button class="btn-icon" onclick="document.getElementById('shortcuts-modal').classList.remove('open')" aria-label="Close">✕</button>
        </div>
        <div style="display:grid; grid-template-columns:auto 1fr; gap:0.5rem 1rem; font-size:var(--text-xs); align-items:center;">
          <kbd style="font-family:var(--font-mono); background:var(--bg-app); border:1px solid var(--border-base); padding:2px 6px; border-radius:3px;">?</kbd>
          <span>Open this shortcut reference sheet</span>

          <kbd style="font-family:var(--font-mono); background:var(--bg-app); border:1px solid var(--border-base); padding:2px 6px; border-radius:3px;">↑ / ↓</kbd>
          <span>Navigate review queue rows</span>

          <kbd style="font-family:var(--font-mono); background:var(--bg-app); border:1px solid var(--border-base); padding:2px 6px; border-radius:3px;">Enter</kbd>
          <span>Open evidence for selected unit</span>

          <kbd style="font-family:var(--font-mono); background:var(--bg-app); border:1px solid var(--border-base); padding:2px 6px; border-radius:3px;">[ / ]</kbd>
          <span>Step cycle backward / forward</span>

          <kbd style="font-family:var(--font-mono); background:var(--bg-app); border:1px solid var(--border-base); padding:2px 6px; border-radius:3px;">Space</kbd>
          <span>Play / pause replay stream</span>

          <kbd style="font-family:var(--font-mono); background:var(--bg-app); border:1px solid var(--border-base); padding:2px 6px; border-radius:3px;">1 - 4</kbd>
          <span>Switch dataset FD001 - FD004</span>

          <kbd style="font-family:var(--font-mono); background:var(--bg-app); border:1px solid var(--border-base); padding:2px 6px; border-radius:3px;">f / u / a</kbd>
          <span>Queue filter: Flagged / Unreviewed / All</span>

          <kbd style="font-family:var(--font-mono); background:var(--bg-app); border:1px solid var(--border-base); padding:2px 6px; border-radius:3px;">t</kbd>
          <span>Toggle dark / light theme</span>

          <kbd style="font-family:var(--font-mono); background:var(--bg-app); border:1px solid var(--border-base); padding:2px 6px; border-radius:3px;">s</kbd>
          <span>Toggle settings drawer</span>

          <kbd style="font-family:var(--font-mono); background:var(--bg-app); border:1px solid var(--border-base); padding:2px 6px; border-radius:3px;">Esc</kbd>
          <span>Close open modal / drawer</span>
        </div>
      </div>
    `;
    document.body.appendChild(modal);
    modal.addEventListener('click', (e) => {
      if (e.target === modal) modal.classList.remove('open');
    });
  }

  document.addEventListener('DOMContentLoaded', () => {
    createShortcutsModal();

    window.addEventListener('keydown', (e) => {
      if (['input', 'textarea', 'select'].includes(document.activeElement.tagName.toLowerCase())) {
        if (e.key === 'Escape') {
          document.activeElement.blur();
        }
        return;
      }

      if (e.key === '?') {
        const modal = document.getElementById('shortcuts-modal');
        if (modal) modal.classList.toggle('open');
      } else if (e.key === 'Escape') {
        const modal = document.getElementById('shortcuts-modal');
        if (modal && modal.classList.contains('open')) modal.classList.remove('open');
        const drawer = document.getElementById('settings-drawer');
        const backdrop = document.getElementById('drawer-backdrop');
        if (drawer && drawer.classList.contains('open')) {
          drawer.classList.remove('open');
          if (backdrop) backdrop.classList.remove('open');
        }
        const polModal = document.getElementById('policy-compare-modal');
        if (polModal) polModal.style.display = 'none';
      } else if (e.key === 't' || e.key === 'T') {
        const currentTheme = EvidentPrefs.get('theme');
        const nextTheme = currentTheme === 'dark' ? 'light' : 'dark';
        EvidentPrefs.set('theme', nextTheme);
        EvidentPrefs.apply();
      } else if (e.key === 's' || e.key === 'S') {
        const drawer = document.getElementById('settings-drawer');
        const backdrop = document.getElementById('drawer-backdrop');
        if (drawer) {
          const isOpen = drawer.classList.toggle('open');
          if (backdrop) backdrop.classList.toggle('open', isOpen);
        }
      } else if (['1', '2', '3', '4'].includes(e.key)) {
        const ds = `FD00${e.key}`;
        const url = new URL(window.location.href);
        url.searchParams.set('subset', ds);
        window.location.href = url.toString();
      } else if (['f', 'F'].includes(e.key) && window.evidentQueue) {
        window.evidentQueue.setFilter('flagged');
      } else if (['u', 'U'].includes(e.key) && window.evidentQueue) {
        window.evidentQueue.setFilter('unreviewed');
      } else if (['a', 'A'].includes(e.key) && window.evidentQueue) {
        window.evidentQueue.setFilter('all');
      }
    });
  });
})();

