/**
 * EVIDENT - Browser Preferences Management (Local-only, key prefix: evident.prefs.)
 * Personalization changes how an engineer sees and works - never changes models or predictions.
 */
(function() {
  const PREFIX = 'evident.prefs.';
  
  const DEFAULTS = {
    theme: 'dark', // 'dark', 'light', 'system'
    density: 'comfortable', // 'compact', 'comfortable'
    textSize: 'normal', // 'normal', 'large'
    reducedEffects: false,
    defaultSubset: 'FD001',
    defaultCycle: 50,
    queueFilter: 'all', // 'all', 'flagged', 'unreviewed', 'watched'
    queueSort: 'q50', // 'q50', 'unit', 'cycle'
    startingView: 'engineer', // 'engineer', 'manager', 'analyst'
    defaultPolicy: 'v1', // 'v1', 'v2'
    watchedUnits: {}, // e.g. { "FD001": [4, 12] }
    pinnedSensors: {}, // e.g. { "FD001": ["s_2", "s_3"] }
    visibleColumns: {
      cycle: true,
      q50: true,
      band: true,
      flag: true,
      caveats: true,
      review: true
    }
  };

  const Prefs = {
    get(key) {
      try {
        const val = localStorage.getItem(PREFIX + key);
        if (val === null) return DEFAULTS[key];
        return JSON.parse(val);
      } catch (e) {
        return DEFAULTS[key];
      }
    },

    set(key, val) {
      try {
        localStorage.setItem(PREFIX + key, JSON.stringify(val));
      } catch (e) {
        console.warn('LocalStorage unavailable');
      }
    },

    reset() {
      try {
        Object.keys(DEFAULTS).forEach(k => localStorage.removeItem(PREFIX + k));
      } catch (e) {}
      this.apply();
      window.location.reload();
    },

    isWatched(subset, unit) {
      const watched = this.get('watchedUnits') || {};
      const list = watched[subset] || [];
      return list.includes(Number(unit));
    },

    toggleWatch(subset, unit) {
      const watched = this.get('watchedUnits') || {};
      const list = watched[subset] || [];
      const num = Number(unit);
      const idx = list.indexOf(num);
      if (idx > -1) {
        list.splice(idx, 1);
      } else {
        list.push(num);
      }
      watched[subset] = list;
      this.set('watchedUnits', watched);
      return idx === -1;
    },

    getPinnedSensors(subset) {
      const all = this.get('pinnedSensors') || {};
      return all[subset] || [];
    },

    togglePinSensor(subset, channel) {
      const all = this.get('pinnedSensors') || {};
      const list = (all[subset] || []).slice();
      const idx = list.indexOf(channel);
      if (idx > -1) {
        list.splice(idx, 1);
      } else {
        list.push(channel);
      }
      all[subset] = list;
      this.set('pinnedSensors', all);
      return idx === -1;
    },

    applyColumnVisibility() {
      const cols = this.get('visibleColumns') || DEFAULTS.visibleColumns;
      ['cycle', 'q50', 'band', 'flag', 'caveats', 'review'].forEach(col => {
        const isVis = cols[col] !== false;
        document.querySelectorAll(`.col-${col}`).forEach(el => {
          el.style.display = isVis ? '' : 'none';
        });
      });
    },

    apply() {
      const html = document.documentElement;
      
      // Theme
      let theme = this.get('theme');
      const urlParams = new URLSearchParams(window.location.search);
      if (urlParams.get('theme')) {
        theme = urlParams.get('theme');
      } else if (theme === 'system') {
        const darkMq = window.matchMedia('(prefers-color-scheme: dark)');
        theme = darkMq.matches ? 'dark' : 'light';
      }
      html.setAttribute('data-theme', theme);

      // Density
      const density = urlParams.get('density') || this.get('density');
      html.setAttribute('data-density', density);

      // Text Size
      const textSize = urlParams.get('text_size') || this.get('textSize');
      html.setAttribute('data-text-size', textSize);

      // Reduced Effects
      const reduced = urlParams.get('reduced') === '1' || urlParams.get('reduced') === 'true' || this.get('reducedEffects');
      html.setAttribute('data-reduced-effects', reduced ? 'true' : 'false');

      // Column visibility
      this.applyColumnVisibility();
    }
  };

  // Apply on load immediately
  Prefs.apply();

  // Watch system color scheme changes
  window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', () => {
    if (Prefs.get('theme') === 'system') {
      Prefs.apply();
    }
  });

  window.EvidentPrefs = Prefs;
})();
