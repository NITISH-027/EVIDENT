/**
 * EVIDENT - Interactive Vanilla SVG Charts
 * High precision, zero dependencies, print and screen-reader accessible.
 */
(function() {
  const EvidentChart = {
    renderRulChart(container, options) {
      if (!container) return;
      const {
        series = [],
        currentCycle = 1,
        firstFlagCycle = null,
        horizon = 30,
        showTruth = false,
        bandV2 = false,
        v1Coverage = 54,
        v2Coverage = null,
        v1Width = null,
        v2Width = null,
        onCycleSelect = null
      } = options;

      if (!series.length) {
        container.innerHTML = '<div class="honesty-notice" style="padding: 2rem; text-align: center;">No replay series data available.</div>';
        return;
      }

      const rect = container.getBoundingClientRect();
      const width = Math.max(rect.width || 600, 320);
      const height = 300;
      const margin = { top: 24, right: 30, bottom: 36, left: 45 };
      const plotW = width - margin.left - margin.right;
      const plotH = height - margin.top - margin.bottom;

      // Compute domains
      const minCycle = Math.min(...series.map(d => d.cycle));
      const maxCycle = Math.max(...series.map(d => d.cycle));
      
      let maxVal = Math.max(
        horizon * 1.5,
        ...series.map(d => Math.max(d.q90 || 0, (bandV2 && d.q90_v2) || 0, (showTruth && d.true_rul) || 0))
      );
      maxVal = Math.ceil(maxVal * 1.08);

      const xScale = c => margin.left + ((c - minCycle) / Math.max(1, maxCycle - minCycle)) * plotW;
      const yScale = v => margin.top + plotH - (Math.max(0, v) / maxVal) * plotH;

      // Accessible screen-reader text summary
      const curPoint = series.find(d => d.cycle === currentCycle) || series[series.length - 1];
      const curQ50 = curPoint ? curPoint.q50 : '-';
      const curQ10 = curPoint ? (bandV2 && curPoint.q10_v2 != null ? curPoint.q10_v2 : curPoint.q10) : '-';
      const curQ90 = curPoint ? (bandV2 && curPoint.q90_v2 != null ? curPoint.q90_v2 : curPoint.q90) : '-';
      const curFlag = curPoint && curPoint.flag ? 'FLAGGED' : 'not flagged';
      const srText = `RUL trajectory from cycle ${minCycle} to ${maxCycle}. Current cycle ${currentCycle}: predicted median ${curQ50} cycles, band [${curQ10}, ${curQ90}], status ${curFlag}. Warning horizon ${horizon} cycles.`;

      // Build SVG
      let svg = `
        <div class="sr-only" style="position: absolute; width: 1px; height: 1px; overflow: hidden; clip: rect(0,0,0,0);">${srText}</div>
        <svg class="chart-svg" viewBox="0 0 ${width} ${height}" preserveAspectRatio="none" role="img" aria-label="RUL trajectory chart">
          <defs>
            <clipPath id="plot-clip"><rect x="${margin.left}" y="${margin.top}" width="${plotW}" height="${plotH}" /></clipPath>
          </defs>
          <!-- Grid Lines -->
      `;

      // Horizontal grid & ticks
      const yTicks = [0, horizon, Math.round(maxVal / 2), Math.round(maxVal)];
      const uniqueYTicks = [...new Set(yTicks)].sort((a,b) => a - b);
      uniqueYTicks.forEach(t => {
        const y = yScale(t);
        svg += `
          <line x1="${margin.left}" y1="${y}" x2="${margin.left + plotW}" y2="${y}" stroke="var(--chart-grid)" stroke-width="1" />
          <text x="${margin.left - 8}" y="${y + 4}" fill="var(--chart-axis)" font-size="10" font-family="var(--font-mono)" text-anchor="end">${t}</text>
        `;
      });

      // Warning threshold line
      const threshY = yScale(horizon);
      svg += `
        <line x1="${margin.left}" y1="${threshY}" x2="${margin.left + plotW}" y2="${threshY}" stroke="var(--chart-threshold)" stroke-width="1.2" stroke-dasharray="3,3" />
        <text x="${margin.left + plotW - 6}" y="${threshY - 4}" fill="var(--chart-threshold)" font-size="10" font-family="var(--font-mono)" text-anchor="end">horizon ${horizon}</text>
      `;

      // Vertical x-ticks
      const xTickCount = Math.min(6, maxCycle - minCycle);
      const xStep = Math.max(1, Math.round((maxCycle - minCycle) / xTickCount));
      for (let c = minCycle; c <= maxCycle; c += xStep) {
        const x = xScale(c);
        svg += `
          <line x1="${x}" y1="${margin.top + plotH}" x2="${x}" y2="${margin.top + plotH + 5}" stroke="var(--chart-axis)" stroke-width="1" />
          <text x="${x}" y="${margin.top + plotH + 18}" fill="var(--chart-axis)" font-size="10" font-family="var(--font-mono)" text-anchor="middle">${c}</text>
        `;
      }

      svg += `<g clip-path="url(#plot-clip)">`;

      // 1. Uncertainty Band
      if (bandV2) {
        // v2 Calibrated Band
        const upperV2 = series.map(d => `${xScale(d.cycle)},${yScale(d.q90_v2 != null ? d.q90_v2 : d.q90)}`).join(' ');
        const lowerV2 = [...series].reverse().map(d => `${xScale(d.cycle)},${yScale(d.q10_v2 != null ? d.q10_v2 : d.q10)}`).join(' ');
        svg += `<polygon points="${upperV2} ${lowerV2}" fill="var(--chart-band-v2)" />`;

        // Also subtle outline of v1 for comparison
        const upperV1 = series.map(d => `${xScale(d.cycle)},${yScale(d.q90)}`).join(' ');
        const lowerV1 = [...series].reverse().map(d => `${xScale(d.cycle)},${yScale(d.q10)}`).join(' ');
        svg += `<polygon points="${upperV1} ${lowerV1}" fill="var(--chart-band)" opacity="0.6" stroke="var(--border-muted)" stroke-width="0.5" stroke-dasharray="2,2" />`;
      } else {
        // v1 Nominal 80% Band
        const upper = series.map(d => `${xScale(d.cycle)},${yScale(d.q90)}`).join(' ');
        const lower = [...series].reverse().map(d => `${xScale(d.cycle)},${yScale(d.q10)}`).join(' ');
        svg += `<polygon points="${upper} ${lower}" fill="var(--chart-band)" stroke="var(--chart-median)" stroke-width="0.5" stroke-opacity="0.4" />`;
      }

      // 2. Retrospective Ground Truth (if toggled)
      if (showTruth) {
        const truthPts = series.filter(d => d.true_rul != null).map(d => `${xScale(d.cycle)},${yScale(d.true_rul)}`).join(' ');
        if (truthPts) {
          svg += `<polyline points="${truthPts}" fill="none" stroke="var(--chart-truth)" stroke-width="1.8" stroke-dasharray="4,3" />`;
        }
      }

      // 3. Predicted Median Line (q50)
      const medianPts = series.map(d => `${xScale(d.cycle)},${yScale(d.q50)}`).join(' ');
      svg += `<polyline points="${medianPts}" fill="none" stroke="var(--chart-median)" stroke-width="2.2" />`;

      // 4. First-flag Marker (amber vertical line)
      if (firstFlagCycle && firstFlagCycle >= minCycle && firstFlagCycle <= maxCycle) {
        const fx = xScale(firstFlagCycle);
        svg += `
          <line x1="${fx}" y1="${margin.top}" x2="${fx}" y2="${margin.top + plotH}" stroke="var(--chart-flag-marker)" stroke-width="1.5" />
          <rect x="${fx - 28}" y="${margin.top + 2}" width="56" height="15" fill="var(--bg-card)" stroke="var(--chart-flag-marker)" rx="2" />
          <text x="${fx}" y="${margin.top + 13}" fill="var(--flag-amber)" font-size="9" font-family="var(--font-mono)" font-weight="bold" text-anchor="middle">FIRST FLAG</text>
        `;
      }

      // 5. Current Cycle Cursor Line
      if (currentCycle >= minCycle && currentCycle <= maxCycle) {
        const cx = xScale(currentCycle);
        svg += `
          <line x1="${cx}" y1="${margin.top}" x2="${cx}" y2="${margin.top + plotH}" stroke="var(--text-main)" stroke-width="1.2" stroke-dasharray="2,2" opacity="0.8" />
          <circle cx="${cx}" cy="${yScale(curPoint.q50)}" r="4.5" fill="var(--chart-median)" stroke="var(--bg-app)" stroke-width="2" />
        `;
      }

      // Interactive hover overlay rect
      svg += `
        <rect id="rul-hover-rect" x="${margin.left}" y="${margin.top}" width="${plotW}" height="${plotH}" fill="transparent" style="cursor: crosshair;" />
      `;

      svg += `</g>`;

      // Axis labels
      svg += `
        <text x="${margin.left + plotW / 2}" y="${height - 6}" fill="var(--text-dim)" font-size="10" text-anchor="middle">Cycle</text>
        <text x="12" y="${margin.top + plotH / 2}" fill="var(--text-dim)" font-size="10" text-anchor="middle" transform="rotate(-90 12 ${margin.top + plotH / 2})">RUL (cycles)</text>
      `;

      svg += `</svg>`;
      container.innerHTML = svg;

      // Setup hover interactions
      const hoverRect = container.querySelector('#rul-hover-rect');
      const readout = document.getElementById('rul-chart-readout');
      if (hoverRect && readout) {
        hoverRect.addEventListener('mousemove', (e) => {
          const cRect = hoverRect.getBoundingClientRect();
          const mouseX = e.clientX - cRect.left;
          const pct = Math.max(0, Math.min(1, mouseX / cRect.width));
          const targetCycle = Math.round(minCycle + pct * (maxCycle - minCycle));
          const point = series.find(d => d.cycle === targetCycle) || series[series.length - 1];
          if (point) {
            const q10Val = bandV2 && point.q10_v2 != null ? point.q10_v2 : point.q10;
            const q90Val = bandV2 && point.q90_v2 != null ? point.q90_v2 : point.q90;
            readout.innerHTML = `
              <span>Cycle: <b>${point.cycle}</b></span>
              <span>Median q50: <b>${point.q50}</b></span>
              <span>Band: <b>[${q10Val}, ${q90Val}]</b></span>
              ${point.flag ? '<span class="chip chip-flag">FLAG</span>' : '<span class="chip chip-noflag">no flag</span>'}
              ${showTruth && point.true_rul != null ? `<span style="color:var(--chart-truth)">True RUL: <b>${point.true_rul.toFixed(1)}</b></span>` : ''}
            `;
          }
        });

        hoverRect.addEventListener('click', (e) => {
          const cRect = hoverRect.getBoundingClientRect();
          const mouseX = e.clientX - cRect.left;
          const pct = Math.max(0, Math.min(1, mouseX / cRect.width));
          const targetCycle = Math.round(minCycle + pct * (maxCycle - minCycle));
          if (onCycleSelect && typeof onCycleSelect === 'function') {
            onCycleSelect(targetCycle);
          }
        });

        hoverRect.addEventListener('mouseleave', () => {
          // Reset to current cycle readout
          const cur = series.find(d => d.cycle === currentCycle) || series[series.length - 1];
          if (cur) {
            const q10Val = bandV2 && cur.q10_v2 != null ? cur.q10_v2 : cur.q10;
            const q90Val = bandV2 && cur.q90_v2 != null ? cur.q90_v2 : cur.q90;
            readout.innerHTML = `
              <span>Cycle: <b>${cur.cycle}</b></span>
              <span>Median q50: <b>${cur.q50}</b></span>
              <span>Band: <b>[${q10Val}, ${q90Val}]</b></span>
              ${cur.flag ? '<span class="chip chip-flag">FLAG</span>' : '<span class="chip chip-noflag">no flag</span>'}
              ${showTruth && cur.true_rul != null ? `<span style="color:var(--chart-truth)">True RUL: <b>${cur.true_rul.toFixed(1)}</b></span>` : ''}
            `;
          }
        });
      }
    },

    renderSensors(container, sensorData, subset = null, onTogglePin = null) {
      if (!container || !sensorData || !sensorData.channels) return;
      const { channels, single_regime, caveat } = sensorData;

      // Reorder channels if any are pinned for this subset
      const pinnedList = (window.EvidentPrefs && subset) ? (EvidentPrefs.getPinnedSensors(subset) || []) : [];
      const pinnedChannels = [];
      const unpinnedChannels = [];

      pinnedList.forEach(pName => {
        const found = channels.find(c => c.channel === pName);
        if (found) pinnedChannels.push(found);
      });
      channels.forEach(c => {
        if (!pinnedList.includes(c.channel)) {
          unpinnedChannels.push(c);
        }
      });
      const orderedChannels = [...pinnedChannels, ...unpinnedChannels];

      let html = `<div class="sensors-grid">`;
      orderedChannels.forEach((ch, idx) => {
        const isPinned = pinnedList.includes(ch.channel);
        html += `
          <div class="sensor-card">
            <div class="sensor-card-header">
              <div style="display:flex; align-items:center; gap:var(--space-2);">
                <button class="pin-btn" data-channel="${ch.channel}" title="${isPinned ? 'Unpin channel' : 'Pin channel to show first'}" aria-label="${isPinned ? 'Unpin channel' : 'Pin channel'}">
                  ${isPinned ? '📌' : '📍'}
                </button>
                <span class="sensor-name">Channel: ${ch.channel}</span>
              </div>
              <div style="display:flex; align-items:center; gap:var(--space-1);">
                ${isPinned ? '<span class="chip chip-reviewed" style="font-size:0.65rem;">pinned</span>' : ''}
                <span class="chip chip-caveat" style="font-size:0.65rem;">trend signal</span>
              </div>
            </div>
            <div class="sensor-chart-wrap" id="sensor-chart-${idx}"></div>
          </div>
        `;
      });
      html += `</div>`;
      html += `<div class="honesty-notice" style="margin-top: 0.5rem;">${caveat}</div>`;
      container.innerHTML = html;

      // Bind pin toggle buttons
      container.querySelectorAll('.pin-btn').forEach(btn => {
        btn.addEventListener('click', (e) => {
          e.stopPropagation();
          const chName = btn.getAttribute('data-channel');
          if (window.EvidentPrefs && subset) {
            EvidentPrefs.togglePinSensor(subset, chName);
            EvidentChart.renderSensors(container, sensorData, subset, onTogglePin);
            if (onTogglePin) onTogglePin();
          }
        });
      });

      // Render miniature charts for each channel
      orderedChannels.forEach((ch, idx) => {
        const wrap = container.querySelector(`#sensor-chart-${idx}`);
        if (!wrap) return;
        const rect = wrap.getBoundingClientRect();
        const width = Math.max(rect.width || 240, 180);
        const height = 130;
        const m = { top: 10, right: 10, bottom: 20, left: 35 };
        const pw = width - m.left - m.right;
        const ph = height - m.top - m.bottom;

        const cycles = ch.cycles;
        const minC = Math.min(...cycles);
        const maxC = Math.max(...cycles);

        const allVals = single_regime ? [...ch.readings, ...ch.baseline] : ch.diff;
        const minV = Math.min(...allVals);
        const maxV = Math.max(...allVals);
        const range = Math.max(1e-5, maxV - minV);

        const xS = c => m.left + ((c - minC) / Math.max(1, maxC - minC)) * pw;
        const yS = v => m.top + ph - ((v - minV) / range) * ph;

        let sSvg = `<svg class="chart-svg" viewBox="0 0 ${width} ${height}">`;
        sSvg += `<line x1="${m.left}" y1="${m.top + ph}" x2="${m.left + pw}" y2="${m.top + ph}" stroke="var(--chart-grid)" />`;
        sSvg += `<line x1="${m.left}" y1="${m.top}" x2="${m.left}" y2="${m.top + ph}" stroke="var(--chart-grid)" />`;

        if (single_regime) {
          // Readings & Baseline
          const rPts = ch.readings.map((v, i) => `${xS(cycles[i])},${yS(v)}`).join(' ');
          const bPts = ch.baseline.map((v, i) => `${xS(cycles[i])},${yS(v)}`).join(' ');
          sSvg += `<polyline points="${bPts}" fill="none" stroke="var(--chart-truth)" stroke-width="1.2" />`;
          sSvg += `<polyline points="${rPts}" fill="none" stroke="var(--accent-blue)" stroke-width="1" />`;
        } else {
          // Deviation from matched baseline
          const zeroY = yS(0);
          sSvg += `<line x1="${m.left}" y1="${zeroY}" x2="${m.left + pw}" y2="${zeroY}" stroke="var(--chart-threshold)" stroke-width="0.8" stroke-dasharray="2,2" />`;
          const dPts = ch.diff.map((v, i) => `${xS(cycles[i])},${yS(v)}`).join(' ');
          sSvg += `<polyline points="${dPts}" fill="none" stroke="var(--accent-blue)" stroke-width="1.2" />`;

          // Fallback dots
          if (ch.fallback) {
            ch.fallback.forEach((isFb, i) => {
              if (isFb) {
                sSvg += `<circle cx="${xS(cycles[i])}" cy="${yS(ch.diff[i])}" r="2" fill="var(--chart-truth)" />`;
              }
            });
          }
        }

        sSvg += `<text x="${m.left}" y="${m.top + 8}" fill="var(--text-dim)" font-size="8" font-family="var(--font-mono)">${maxV.toFixed(1)}</text>`;
        sSvg += `<text x="${m.left}" y="${m.top + ph}" fill="var(--text-dim)" font-size="8" font-family="var(--font-mono)">${minV.toFixed(1)}</text>`;
        sSvg += `</svg>`;

        wrap.innerHTML = sSvg;
      });
    }
  };

  window.EvidentChart = EvidentChart;
})();

