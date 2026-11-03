/**
 * EVIDENT - Evidence Panel Controller
 * RUL chart, sensor trend context, replay stepping, scrubber, in-page play loop,
 * policy comparison, and explicit flag-display logging.
 */
(function() {
  class EvidencePanel {
    constructor(options) {
      this.container = options.container;
      this.chartContainer = options.chartContainer;
      this.sensorsContainer = options.sensorsContainer;
      this.scrubber = options.scrubber;
      this.stepPrevBtn = options.stepPrevBtn;
      this.stepNextBtn = options.stepNextBtn;
      this.firstFlagBtn = options.firstFlagBtn;
      this.latestBtn = options.latestBtn;
      this.playBtn = options.playBtn;
      this.speedSelect = options.speedSelect;
      this.truthToggle = options.truthToggle;
      this.bandToggle = options.bandToggle;
      this.compareBtn = options.compareBtn;

      this.subset = options.subset || 'FD001';
      this.version = options.version || 'v1';
      this.unit = options.initialUnit || null;
      this.cycle = options.initialCycle || 50;

      this.unitData = null;
      this.sensorData = null;
      this.isPlaying = false;
      this.playSpeed = 1; // 1x, 2x, 4x
      this.playInterval = null;
      this.showTruth = options.initialTruth || false;
      this.bandV2 = options.initialBandV2 || false;

      if (this.truthToggle) this.truthToggle.checked = this.showTruth;
      if (this.bandToggle) this.bandToggle.checked = this.bandV2;

      this.initEvents();
    }

    initEvents() {
      // Step buttons
      if (this.stepPrevBtn) {
        this.stepPrevBtn.addEventListener('click', () => this.stepCycle(-1));
      }
      if (this.stepNextBtn) {
        this.stepNextBtn.addEventListener('click', () => this.stepCycle(1));
      }
      if (this.firstFlagBtn) {
        this.firstFlagBtn.addEventListener('click', () => {
          if (this.unitData && this.unitData.first_flag_cycle) {
            this.setCycle(this.unitData.first_flag_cycle);
          }
        });
      }
      if (this.latestBtn) {
        this.latestBtn.addEventListener('click', () => {
          if (this.unitData && this.unitData.max_cycle) {
            this.setCycle(this.unitData.max_cycle);
          }
        });
      }

      // Scrubber
      if (this.scrubber) {
        this.scrubber.addEventListener('input', (e) => {
          const c = parseInt(e.target.value, 10);
          this.setCycle(c, false); // don't push state on every micro-scrub
        });
        this.scrubber.addEventListener('change', (e) => {
          const c = parseInt(e.target.value, 10);
          this.setCycle(c, true);
        });
      }

      // Play / Pause stream
      if (this.playBtn) {
        this.playBtn.addEventListener('click', () => this.togglePlay());
      }
      if (this.speedSelect) {
        this.speedSelect.addEventListener('change', (e) => {
          this.playSpeed = parseFloat(e.target.value) || 1;
          if (this.isPlaying) {
            this.startPlayLoop();
          }
        });
      }

      // Truth toggle
      if (this.truthToggle) {
        this.truthToggle.addEventListener('change', (e) => {
          this.showTruth = e.target.checked;
          this.loadUnit(this.unit, this.cycle, true);
        });
      }

      // Band toggle (v1 vs v2 calibrated)
      if (this.bandToggle) {
        this.bandToggle.addEventListener('change', (e) => {
          this.bandV2 = e.target.checked;
          this.loadUnit(this.unit, this.cycle, true);
        });
      }

      // Compare policies button
      if (this.compareBtn) {
        this.compareBtn.addEventListener('click', () => this.showPolicyCompare());
      }

      // Keyboard shortcuts for cycling [ and ]
      window.addEventListener('keydown', (e) => {
        if (['input', 'textarea'].includes(document.activeElement.tagName.toLowerCase())) {
          return;
        }
        if (e.key === '[') {
          e.preventDefault();
          this.stepCycle(-1);
        } else if (e.key === ']') {
          e.preventDefault();
          this.stepCycle(1);
        } else if (e.key === ' ' && this.unit) {
          e.preventDefault();
          this.togglePlay();
        }
      });

      // Browser popstate (Back/Forward)
      window.addEventListener('popstate', (e) => {
        const url = new URL(window.location.href);
        const u = url.searchParams.get('unit') || (url.pathname.startsWith('/unit/') ? url.pathname.split('/')[3] : null);
        const c = url.searchParams.get('cycle');
        const s = url.searchParams.get('subset');
        const v = url.searchParams.get('version');
        const b = url.searchParams.get('band');
        const t = url.searchParams.get('show_truth');

        if (s && s !== this.subset) {
          window.location.reload();
          return;
        }
        if (v && v !== this.version) {
          this.version = v;
        }
        if (b !== null) {
          this.bandV2 = (b === 'v2');
          if (this.bandToggle) this.bandToggle.checked = this.bandV2;
        } else {
          this.bandV2 = false;
          if (this.bandToggle) this.bandToggle.checked = false;
        }
        if (t !== null) {
          this.showTruth = (t === '1' || t === 'true');
          if (this.truthToggle) this.truthToggle.checked = this.showTruth;
        } else {
          this.showTruth = false;
          if (this.truthToggle) this.truthToggle.checked = false;
        }
        const targetUnit = u ? parseInt(u, 10) : this.unit;
        const targetCycle = c ? parseInt(c, 10) : this.cycle;
        if (targetUnit) {
          this.loadUnit(targetUnit, targetCycle, false);
          if (window.evidentQueue && window.evidentQueue.selectUnit) {
            window.evidentQueue.selectUnit(targetUnit, false);
          }
        }
      });
    }

    stepCycle(delta) {
      if (!this.unitData) return;
      const target = Math.max(1, Math.min(this.unitData.max_cycle, this.cycle + delta));
      this.setCycle(target);
    }

    setCycle(cycle, updateUrl = true) {
      this.cycle = cycle;
      if (this.scrubber) {
        this.scrubber.value = cycle;
      }
      const scrubberVal = document.getElementById('scrubber-cycle-val');
      if (scrubberVal) scrubberVal.textContent = cycle;

      this.loadUnit(this.unit, cycle, updateUrl);
    }

    togglePlay() {
      if (this.isPlaying) {
        this.stopPlayLoop();
      } else {
        this.startPlayLoop();
      }
    }

    startPlayLoop() {
      this.stopPlayLoop();
      this.isPlaying = true;
      if (this.playBtn) {
        this.playBtn.innerHTML = '⏸ Pause';
        this.playBtn.classList.add('btn-primary');
      }
      const intervalMs = Math.max(150, Math.round(800 / this.playSpeed));
      this.playInterval = setInterval(() => {
        if (!this.unitData || this.cycle >= this.unitData.max_cycle) {
          this.stopPlayLoop();
          return;
        }
        this.setCycle(this.cycle + 1, false);
      }, intervalMs);
    }

    stopPlayLoop() {
      this.isPlaying = false;
      if (this.playInterval) {
        clearInterval(this.playInterval);
        this.playInterval = null;
      }
      if (this.playBtn) {
        this.playBtn.innerHTML = '▶ Play';
        this.playBtn.classList.remove('btn-primary');
      }
      // Update URL once stopped
      this.updateUrl();
    }

    updateUrl() {
      if (!this.unit) return;
      const url = new URL(window.location.href);
      url.searchParams.set('subset', this.subset);
      url.searchParams.set('unit', this.unit);
      url.searchParams.set('cycle', this.cycle);
      if (this.version === 'v2') {
        url.searchParams.set('version', 'v2');
      } else {
        url.searchParams.delete('version');
      }
      if (this.bandV2) {
        url.searchParams.set('band', 'v2');
      } else {
        url.searchParams.delete('band');
      }
      if (this.showTruth) {
        url.searchParams.set('show_truth', '1');
      } else {
        url.searchParams.delete('show_truth');
      }
      history.pushState({
        unit: this.unit,
        cycle: this.cycle,
        subset: this.subset,
        version: this.version,
        band: this.bandV2 ? 'v2' : 'v1',
        show_truth: this.showTruth ? '1' : '0'
      }, '', url.toString());
    }

    async loadUnit(unit, cycle = null, updateUrl = true) {
      if (!unit) return;
      this.unit = unit;

      let endpoint = `/api/unit/${this.subset}/${unit}?version=${this.version}`;
      if (cycle) endpoint += `&cycle=${cycle}`;
      if (this.bandV2) endpoint += `&band=v2`;
      if (this.showTruth) endpoint += `&show_truth=1`;

      try {
        const resp = await fetch(endpoint);
        if (!resp.ok) {
          throw new Error(`Failed to load unit: ${resp.status}`);
        }
        const data = await resp.json();
        this.unitData = data;
        this.cycle = data.cycle;

        if (updateUrl) {
          this.updateUrl();
        }

        this.renderEvidence(data);

        // Fetch sensor trend context
        this.loadSensors(unit, data.cycle);

        // Check if currently displayed frame is FLAGGED under active policy:
        // Rule: Log flag_shown when displayed on screen in evidence panel!
        if (data.current && data.current.flag) {
          this.notifyFlagShown(this.subset, unit, data.cycle, this.version);
        }
      } catch (err) {
        console.error(err);
        if (this.container) {
          this.container.innerHTML = `
            <div class="card" style="padding: 2rem; text-align: center;">
              <h3>Unable to load unit ${unit}</h3>
              <p class="honesty-notice">${err.message}</p>
              <button class="btn" style="margin-top: 1rem;" onclick="location.reload()">Retry</button>
            </div>
          `;
        }
      }
    }

    async notifyFlagShown(subset, unit, cycle, version) {
      try {
        await fetch('/api/display-flag', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ subset, unit, cycle, version })
        });
      } catch (e) {
        console.warn('Flag display logging failed', e);
      }
    }

    async loadSensors(unit, cycle) {
      if (!this.sensorsContainer) return;
      try {
        const resp = await fetch(`/api/unit/${this.subset}/${unit}/sensors?cycle=${cycle}`);
        if (!resp.ok) throw new Error('Sensor fetch failed');
        const sData = await resp.json();
        this.sensorData = sData;
        EvidentChart.renderSensors(this.sensorsContainer, sData, this.subset);
      } catch (e) {
        // Fallback to PNG chart
        this.sensorsContainer.innerHTML = `
          <img src="/chart/sensors/${this.subset}/${unit}/${cycle}.png${this.showTruth ? '?show_truth=1' : ''}" style="max-width:100%; border-radius: 4px;" alt="Sensor trend signals" />
        `;
      }
    }

    renderEvidence(data) {
      // 1. Header elements
      const idEl = document.getElementById('evidence-unit-id');
      const cycleMetaEl = document.getElementById('evidence-cycle-meta');
      const flagNoticeEl = document.getElementById('evidence-flag-notice');
      const policyBadgeEl = document.getElementById('evidence-policy-badge');

      if (idEl) idEl.textContent = `${data.subset} Unit ${data.unit}`;
      if (cycleMetaEl) cycleMetaEl.textContent = `cycle ${data.cycle} of ${data.max_cycle}`;

      // Scrubber limits
      if (this.scrubber) {
        this.scrubber.min = 1;
        this.scrubber.max = data.max_cycle;
        this.scrubber.value = data.cycle;
      }
      const scrubberVal = document.getElementById('scrubber-cycle-val');
      if (scrubberVal) scrubberVal.textContent = data.cycle;

      // Forecast metrics
      const cur = data.current;
      const mQ50 = document.getElementById('metric-q50');
      const mBand = document.getElementById('metric-band');
      const mStatus = document.getElementById('metric-status');
      const mCaveat = document.getElementById('metric-caveat');

      if (mQ50) mQ50.textContent = `${cur.q50.toFixed(1)} cycles`;
      
      const q10 = this.bandV2 && cur.q10_v2 != null ? cur.q10_v2 : cur.q10;
      const q90 = this.bandV2 && cur.q90_v2 != null ? cur.q90_v2 : cur.q90;
      if (mBand) mBand.textContent = `[${q10.toFixed(1)}, ${q90.toFixed(1)}]`;

      if (mStatus) {
        if (cur.flag) {
          mStatus.innerHTML = '<span class="chip chip-flag" style="font-size:0.9rem;">FLAG</span>';
        } else {
          mStatus.innerHTML = '<span class="chip chip-noflag" style="font-size:0.9rem;">no flag</span>';
        }
      }

      if (mCaveat) {
        let cavs = [];
        if (cur.limited) cavs.push('Limited history (<30 cycles)');
        if (cur.envelope) cavs.push('Outside training envelope');
        mCaveat.textContent = cavs.join(' · ') || 'within observed range';
      }

      // Flag text banner
      if (flagNoticeEl) {
        if (cur.flag) {
          flagNoticeEl.innerHTML = `
            <div class="caveat-box" style="border-left-color: var(--flag-amber); background: var(--flag-amber-bg); color: var(--flag-amber);">
              <b>WARNING FLAG: predicted remaining life &le; ${data.horizon} cycles (${data.policy_label}).</b>
              <div>This is a threshold flag, not a failure probability.</div>
            </div>
          `;
        } else {
          flagNoticeEl.innerHTML = `
            <div class="honesty-notice" style="padding: 0.35rem 0;">
              No warning flag at cycle ${data.cycle} under ${data.policy_label}. Quiet display is not a health guarantee.
            </div>
          `;
        }
      }

      // Policy Badge in evidence
      if (policyBadgeEl) {
        policyBadgeEl.className = `policy-badge ${data.exploratory ? 'v2-exploratory' : 'v1-stable'}`;
        policyBadgeEl.textContent = data.exploratory ? `v2 exploratory (${data.policy})` : 'v1 stable';
      }

      // Band note attached to chart
      const bandNoteEl = document.getElementById('chart-band-note');
      if (bandNoteEl) {
        if (this.bandV2 && data.v2band) {
          bandNoteEl.textContent = `v2 calibrated band: empirical tune calibration, no guarantee; post-hoc replay coverage ${data.v2band.coverage}% at width ${data.v2band.width} (exploratory, not a probability)`;
        } else {
          bandNoteEl.textContent = `v1 band: nominal 80%; measured coverage on held-out replay: ${data.cov}% - too narrow as fitted, unvalidated`;
        }
      }

      // Render RUL SVG chart
      EvidentChart.renderRulChart(this.chartContainer, {
        series: data.series,
        currentCycle: data.cycle,
        firstFlagCycle: data.first_flag_cycle,
        horizon: data.horizon,
        showTruth: this.showTruth,
        bandV2: this.bandV2,
        v1Coverage: data.cov,
        v2Coverage: data.v2band ? data.v2band.coverage : null,
        onCycleSelect: (targetCycle) => this.setCycle(targetCycle)
      });

      // Update hidden form inputs for verdict
      const formSubset = document.getElementById('verdict-subset');
      const formUnit = document.getElementById('verdict-unit');
      const formCycle = document.getElementById('verdict-cycle');
      const formPolicy = document.getElementById('verdict-policy');
      const formVersion = document.getElementById('verdict-version');
      if (formSubset) formSubset.value = data.subset;
      if (formUnit) formUnit.value = data.unit;
      if (formCycle) formCycle.value = data.cycle;
      if (formPolicy) formPolicy.value = data.policy;
      if (formVersion) formVersion.value = data.version;

      // Update past verdicts list
      const pastVerdictsEl = document.getElementById('past-verdicts-list');
      if (pastVerdictsEl) {
        if (data.past_verdicts && data.past_verdicts.length) {
          let vHtml = '<ul style="list-style:none; padding:0; display:flex; flex-direction:column; gap:0.25rem;">';
          data.past_verdicts.forEach(v => {
            const vLabel = v.verdict === 'reviewed_warrants_inspection' ? 'Warrants Inspection' : 'Dismissed';
            vHtml += `
              <li style="font-size:0.75rem; color:var(--text-muted); border-bottom:1px solid var(--border-muted); padding-bottom:2px;">
                <b>${vLabel}</b> (engineer opinion) at cycle ${v.cycle} [${v.policy}] · <span style="font-family:var(--font-mono);">${v.ts}</span>
              </li>
            `;
          });
          vHtml += '</ul>';
          pastVerdictsEl.innerHTML = vHtml;
        } else {
          pastVerdictsEl.innerHTML = '<span style="font-size:0.75rem; color:var(--text-dim);">No previous verdicts logged for this unit.</span>';
        }
      }
    }

    async showPolicyCompare() {
      if (!this.unitData) return;
      const modal = document.getElementById('policy-compare-modal');
      const content = document.getElementById('policy-compare-content');
      if (!modal || !content) return;

      modal.style.display = 'flex';
      content.innerHTML = '<div style="padding:2rem; text-align:center;">Loading policy comparison...</div>';

      try {
        const [v1Resp, v2Resp] = await Promise.all([
          fetch(`/api/unit/${this.subset}/${this.unit}?version=v1&cycle=${this.cycle}`),
          fetch(`/api/unit/${this.subset}/${this.unit}?version=v2&cycle=${this.cycle}&band=v2`)
        ]);
        const v1Data = await v1Resp.json();
        const v2Data = await v2Resp.json();

        content.innerHTML = `
          <div class="compare-grid">
            <div class="policy-col">
              <div style="font-weight:700; margin-bottom:0.5rem; color:var(--text-main);">v1 Stable Policy (Primary Baseline)</div>
              <div style="font-size:0.75rem; color:var(--text-muted); margin-bottom:0.5rem;">${v1Data.policy_label}</div>
              <div style="display:flex; flex-direction:column; gap:0.35rem; font-size:0.8rem;">
                <div>Trigger condition: <b>predicted median &le; ${v1Data.horizon} cycles</b></div>
                <div>Flag at cycle ${this.cycle}: ${v1Data.current.flag ? '<span class="chip chip-flag">FLAG</span>' : '<span class="chip chip-noflag">no flag</span>'}</div>
                <div>First alert cycle: <b>${v1Data.first_flag_cycle || 'None'}</b></div>
                <div>Band coverage: <b>${v1Data.cov}%</b> (unvalidated nominal 80%)</div>
              </div>
            </div>
            <div class="policy-col v2">
              <div style="font-weight:700; margin-bottom:0.5rem; color:var(--text-main);">v2 Exploratory Policy (Post-hoc Tune Selected)</div>
              <div style="font-size:0.75rem; color:var(--text-muted); margin-bottom:0.5rem;">${v2Data.policy_label}</div>
              <div style="display:flex; flex-direction:column; gap:0.35rem; font-size:0.8rem;">
                <div>Selected policy: <b>${v2Data.policy}</b></div>
                <div>Flag at cycle ${this.cycle}: ${v2Data.current.flag ? '<span class="chip chip-flag">FLAG</span>' : '<span class="chip chip-noflag">no flag</span>'}</div>
                <div>First alert cycle: <b>${v2Data.first_flag_cycle || 'None'}</b></div>
                <div>Calibrated coverage: <b>${v2Data.v2band ? v2Data.v2band.coverage + '%' : 'N/A'}</b> (post-hoc empirical calibration)</div>
              </div>
            </div>
          </div>
          <div class="honesty-notice" style="margin-top:1rem;">
            v1 above remains the baseline and shipped stable default. All v2 metrics are post-hoc tune-selected references and not validated improvements.
          </div>
        `;
      } catch (err) {
        content.innerHTML = `<div class="honesty-notice">Failed to load comparison: ${err.message}</div>`;
      }
    }
  }

  window.EvidentEvidencePanel = EvidencePanel;
})();

