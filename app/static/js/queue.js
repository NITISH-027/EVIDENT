/**
 * EVIDENT - Review Queue Management
 * Filters, sorting, keyboard navigation, and review status tracking.
 */
(function() {
  class ReviewQueue {
    constructor(options) {
      this.tableBody = options.tableBody;
      this.countSummary = options.countSummary;
      this.searchInput = options.searchInput;
      this.filterButtons = options.filterButtons;
      this.sortSelect = options.sortSelect;
      this.cycleInput = options.cycleInput;
      this.onSelectUnit = options.onSelectUnit;

      this.subset = options.subset || 'FD001';
      this.currentCycle = options.cycle || 50;
      this.version = options.version || 'v1';
      this.activePolicy = options.activePolicy || 'v1_q50';

      this.rows = [];
      this.selectedUnit = options.initialUnit || null;
      this.currentFilter = EvidentPrefs.get('queueFilter') || 'all';
      this.currentSort = EvidentPrefs.get('queueSort') || 'q50';
      this.searchQuery = '';

      if (this.filterButtons) {
        this.filterButtons.forEach(b => {
          b.classList.toggle('active', b.getAttribute('data-filter') === this.currentFilter);
        });
      }

      this.initEvents();
    }

    setRows(rows) {
      this.rows = rows;
      this.render();
    }

    initEvents() {
      // Filter buttons
      if (this.filterButtons) {
        this.filterButtons.forEach(btn => {
          btn.addEventListener('click', (e) => {
            const filter = e.target.getAttribute('data-filter');
            this.setFilter(filter);
          });
        });
      }

      // Search
      if (this.searchInput) {
        this.searchInput.addEventListener('input', (e) => {
          this.searchQuery = e.target.value.trim().toLowerCase();
          this.render();
        });
      }

      // Sort
      if (this.sortSelect) {
        this.sortSelect.addEventListener('change', (e) => {
          this.currentSort = e.target.value;
          EvidentPrefs.set('queueSort', this.currentSort);
          this.render();
        });
      }

      // Cycle input
      if (this.cycleInput) {
        this.cycleInput.addEventListener('change', (e) => {
          const val = parseInt(e.target.value, 10);
          if (!isNaN(val) && val > 0) {
            this.changeCycle(val);
          }
        });
      }

      // Keyboard navigation on queue table
      window.addEventListener('keydown', (e) => {
        if (['input', 'textarea'].includes(document.activeElement.tagName.toLowerCase())) {
          return;
        }
        if (e.key === 'ArrowDown') {
          e.preventDefault();
          this.stepQueueSelection(1);
        } else if (e.key === 'ArrowUp') {
          e.preventDefault();
          this.stepQueueSelection(-1);
        } else if (e.key === 'Enter') {
          if (this.selectedUnit && this.onSelectUnit) {
            e.preventDefault();
            this.onSelectUnit(this.selectedUnit, this.currentCycle);
          }
        }
      });
    }

    setFilter(filter) {
      this.currentFilter = filter;
      EvidentPrefs.set('queueFilter', filter);
      if (this.filterButtons) {
        this.filterButtons.forEach(b => {
          b.classList.toggle('active', b.getAttribute('data-filter') === filter);
        });
      }
      this.render();
    }

    changeCycle(cycle) {
      this.currentCycle = cycle;
      const url = new URL(window.location.href);
      url.searchParams.set('cycle', cycle);
      window.location.href = url.toString();
    }

    getFilteredRows() {
      return this.rows.filter(r => {
        // Search
        if (this.searchQuery && !String(r.unit).includes(this.searchQuery)) {
          return false;
        }

        // Watched
        const isWatched = EvidentPrefs.isWatched(this.subset, r.unit);
        if (this.currentFilter === 'watched') {
          return isWatched;
        }

        // Flagged
        if (this.currentFilter === 'flagged') {
          return r.flag;
        }

        // Unreviewed
        if (this.currentFilter === 'unreviewed') {
          const isReviewedHere = r.review && r.review.status === 'reviewed_at_this_cycle';
          return !isReviewedHere;
        }

        return true; // 'all'
      });
    }

    getSortedRows(rows) {
      const sorted = [...rows];
      if (this.currentSort === 'unit') {
        sorted.sort((a, b) => a.unit - b.unit);
      } else if (this.currentSort === 'cycle') {
        sorted.sort((a, b) => b.cycle - a.cycle);
      } else {
        // default q50: flagged units first (flagged & unreviewed at top), then q50 ascending
        sorted.sort((a, b) => {
          const aFlag = a.flag ? 1 : 0;
          const bFlag = b.flag ? 1 : 0;
          if (aFlag !== bFlag) return bFlag - aFlag;
          const aUnrev = !(a.review && a.review.status === 'reviewed_at_this_cycle') ? 1 : 0;
          const bUnrev = !(b.review && b.review.status === 'reviewed_at_this_cycle') ? 1 : 0;
          if (aUnrev !== bUnrev) return bUnrev - aUnrev;
          return a.q50 - b.q50;
        });
      }
      return sorted;
    }

    render() {
      const filtered = this.getFilteredRows();
      const sorted = this.getSortedRows(filtered);

      // Counts
      const totalUnits = this.rows.length;
      const flaggedCount = this.rows.filter(r => r.flag).length;
      const unreviewedCount = this.rows.filter(r => !(r.review && r.review.status === 'reviewed_at_this_cycle')).length;

      if (this.countSummary) {
        this.countSummary.textContent = `${flaggedCount} flagged, ${unreviewedCount} unreviewed of ${totalUnits} units at cycle ${this.currentCycle}`;
      }

      if (!this.tableBody) return;

      if (sorted.length === 0) {
        this.tableBody.innerHTML = `
          <tr>
            <td colspan="7" style="padding: 2rem; text-align: center; color: var(--text-dim);">
              No units match the current filter (${this.currentFilter}).
              <button class="btn" style="margin-left: 0.5rem;" onclick="window.evidentQueue.setFilter('all')">Show All</button>
            </td>
          </tr>
        `;
        return;
      }

      let html = '';
      sorted.forEach(r => {
        const isSel = Number(r.unit) === Number(this.selectedUnit);
        const isWatched = EvidentPrefs.isWatched(this.subset, r.unit);

        // Review status badge
        let revBadge = '<span class="chip chip-unreviewed">Unreviewed</span>';
        if (r.review) {
          if (r.review.status === 'reviewed_at_this_cycle') {
            revBadge = `<span class="chip chip-reviewed" title="Reviewed at this cycle (${r.review.ts})">Reviewed (${r.review.verdict === 'reviewed_warrants_inspection' ? 'warrants' : 'dismissed'})</span>`;
          } else if (r.review.status === 'reviewed_earlier') {
            revBadge = `<span class="chip chip-reviewed-earlier" title="Reviewed at earlier cycle ${r.review.cycle}">Reviewed earlier (cycle ${r.review.cycle})</span>`;
          } else if (r.review.status === 'reviewed_other_policy') {
            revBadge = `<span class="chip chip-other-pol" title="Reviewed under other policy ${r.review.policy}">Reviewed under ${r.review.policy}</span>`;
          }
        }

        // Caveat chips
        let caveats = '';
        if (r.limited) caveats += '<span class="chip chip-caveat">limited history</span> ';
        if (r.envelope) caveats += '<span class="chip chip-caveat">outside envelope</span>';
        if (!caveats) caveats = '<span style="color:var(--text-dim);">-</span>';

        // Flag chip
        const flagCell = r.flag 
          ? '<span class="chip chip-flag">FLAG</span>' 
          : '<span class="chip chip-noflag">no flag</span>';

        html += `
          <tr class="queue-row ${isSel ? 'selected' : ''}" data-unit="${r.unit}" tabindex="0" role="row">
            <td class="col-unit">
              <button class="watch-btn btn-icon" data-unit="${r.unit}" title="${isWatched ? 'Unwatch' : 'Watch'}" style="padding: 2px 4px; font-size: 11px;">
                ${isWatched ? '★' : '☆'}
              </button>
              <span style="font-family: var(--font-mono); font-weight: 600;">${r.unit}</span>
            </td>
            <td class="col-cycle">${r.cycle}</td>
            <td class="col-q50" style="font-family: var(--font-mono); font-weight:600;">${r.q50.toFixed(1)}</td>
            <td class="col-band" style="font-family: var(--font-mono); color: var(--text-muted);">${r.band}</td>
            <td class="col-flag">${flagCell}</td>
            <td class="col-caveats">${caveats}</td>
            <td class="col-review">${revBadge}</td>
          </tr>
        `;
      });

      this.tableBody.innerHTML = html;

      // Apply column visibility
      if (window.EvidentPrefs && EvidentPrefs.applyColumnVisibility) {
        EvidentPrefs.applyColumnVisibility();
      }

      // Bind row clicks
      this.tableBody.querySelectorAll('.queue-row').forEach(row => {
        row.addEventListener('click', (e) => {
          if (e.target.closest('.watch-btn')) return;
          const u = parseInt(row.getAttribute('data-unit'), 10);
          this.selectUnit(u);
        });

        row.addEventListener('keydown', (e) => {
          if (e.key === 'Enter') {
            const u = parseInt(row.getAttribute('data-unit'), 10);
            this.selectUnit(u);
          }
        });
      });

      // Bind watch toggles
      this.tableBody.querySelectorAll('.watch-btn').forEach(btn => {
        btn.addEventListener('click', (e) => {
          e.stopPropagation();
          const u = btn.getAttribute('data-unit');
          EvidentPrefs.toggleWatch(this.subset, u);
          this.render();
        });
      });
    }

    selectUnit(unit, triggerCallback = true) {
      this.selectedUnit = unit;
      this.tableBody.querySelectorAll('.queue-row').forEach(r => {
        r.classList.toggle('selected', Number(r.getAttribute('data-unit')) === Number(unit));
      });
      if (triggerCallback && this.onSelectUnit) {
        this.onSelectUnit(unit, this.currentCycle);
      }
    }

    stepQueueSelection(delta) {
      const filtered = this.getSortedRows(this.getFilteredRows());
      if (!filtered.length) return;
      const curIdx = filtered.findIndex(r => Number(r.unit) === Number(this.selectedUnit));
      let nextIdx = curIdx + delta;
      if (nextIdx < 0) nextIdx = 0;
      if (nextIdx >= filtered.length) nextIdx = filtered.length - 1;
      this.selectUnit(filtered[nextIdx].unit);
      
      // Scroll into view
      const rowEl = this.tableBody.querySelector(`[data-unit="${filtered[nextIdx].unit}"]`);
      if (rowEl) rowEl.scrollIntoView({ block: 'nearest' });
    }

    markUnitReviewed(unit, cycle, verdict, policy) {
      const row = this.rows.find(r => Number(r.unit) === Number(unit));
      if (row) {
        row.review = {
          status: 'reviewed_at_this_cycle',
          verdict: verdict,
          cycle: cycle,
          policy: policy,
          ts: new Date().toISOString().slice(0, 19)
        };
        this.render();
      }
    }
  }

  window.EvidentReviewQueue = ReviewQueue;
})();
