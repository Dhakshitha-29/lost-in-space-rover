/**
 * LOST IN SPACE: ROVER MISSION CONTROL DASHBOARD (Phase 6)
 * Vanilla JavaScript Telemetry, Canvas Rendering, and Simulation Control
 */

(function () {
    'use strict';

    // Application State
    let isRunning = false;
    let timerId = null;
    let currentSpeedMs = 250;
    let latestState = null;
    let isStepping = false;

    // DOM Elements
    const elements = {
        // Headers & Badges
        missionStatusBadge: document.getElementById('mission-status-badge'),
        inputSeed: document.getElementById('input-seed'),
        selectSpeed: document.getElementById('select-speed'),
        toggleDynamicObs: document.getElementById('toggle-dynamic-obs'),

        // Controls
        btnStart: document.getElementById('btn-start'),
        btnPause: document.getElementById('btn-pause'),
        btnResume: document.getElementById('btn-resume'),
        btnStep: document.getElementById('btn-step'),
        btnReset: document.getElementById('btn-reset'),

        // Telemetry Ticker
        stepCount: document.getElementById('telemetry-step-count'),
        coverage: document.getElementById('telemetry-coverage'),
        replans: document.getElementById('telemetry-replans'),

        // Energy KPI
        energyNumeric: document.getElementById('energy-numeric'),
        energyGaugeFill: document.getElementById('energy-gauge-fill'),
        energyPercentage: document.getElementById('energy-percentage'),
        energyUsedLabel: document.getElementById('energy-used-label'),

        // Position KPI
        roverPosDisplay: document.getElementById('rover-pos-display'),
        returnCostVal: document.getElementById('return-cost-val'),
        safetyBufferVal: document.getElementById('safety-buffer-val'),
        baseStatusTag: document.getElementById('base-status-tag'),

        // Science Data KPI
        dataTotalYield: document.getElementById('data-total-yield'),
        dataCollectedVal: document.getElementById('data-collected-val'),
        dataUploadedVal: document.getElementById('data-uploaded-val'),
        dataUnuploadedVal: document.getElementById('data-unuploaded-val'),
        dataLostVal: document.getElementById('data-lost-val'),

        // Goal Directive KPI
        goalTypeBadge: document.getElementById('goal-type-badge'),
        goalTargetDesc: document.getElementById('goal-target-desc'),
        goalUtilityVal: document.getElementById('goal-utility-val'),
        goalCostVal: document.getElementById('goal-cost-val'),

        // Map Meta Badges
        gtDynamicCount: document.getElementById('gt-dynamic-count'),
        beliefKnownCount: document.getElementById('belief-known-count'),

        // Canvases
        canvasGT: document.getElementById('canvas-ground-truth'),
        canvasBelief: document.getElementById('canvas-belief-map'),
        tooltipGT: document.getElementById('tooltip-ground-truth'),
        tooltipBelief: document.getElementById('tooltip-belief-map'),

        // Terrain Costs Table
        costFlatEst: document.getElementById('cost-flat-est'),
        costFlatAct: document.getElementById('cost-flat-act'),
        costFlatAcc: document.getElementById('cost-flat-acc'),
        costRoughEst: document.getElementById('cost-rough-est'),
        costRoughAct: document.getElementById('cost-rough-act'),
        costRoughAcc: document.getElementById('cost-rough-acc'),
        costMountainEst: document.getElementById('cost-mountain-est'),
        costMountainAct: document.getElementById('cost-mountain-act'),
        costMountainAcc: document.getElementById('cost-mountain-acc'),

        // Logs
        logCountBadge: document.getElementById('log-count-badge'),
        decisionLogContainer: document.getElementById('decision-log-container'),

        // Scientific Data Collection & Upload History
        dataHistoryCard: document.getElementById('card-data-history'),
        dataHistoryTbody: document.getElementById('data-history-tbody'),
        dataHistoryEmptyMsg: document.getElementById('data-history-empty-message'),
        pillValCollected: document.getElementById('pill-val-collected'),
        pillValUploaded: document.getElementById('pill-val-uploaded'),
        pillValPending: document.getElementById('pill-val-pending'),
        pillValLost: document.getElementById('pill-val-lost'),

        // Phase 7: Strategy Evaluation
        strategyEvalCard: document.getElementById('card-strategy-eval'),
        strategyEvalTbody: document.getElementById('strategy-eval-tbody'),
        evalBenchmarkInfo: document.getElementById('eval-benchmark-info'),
        btnRefreshEval: document.getElementById('btn-refresh-eval'),
        evalEmptyMsg: document.getElementById('eval-empty-message'),
    };

    const ctxGT = elements.canvasGT.getContext('2d');
    const ctxBelief = elements.canvasBelief.getContext('2d');

    // Color Palette Constants
    const COLORS = {
        bg: '#04060b',
        flat: '#1e3a5f',
        rough: '#b45309',
        hill: '#7c3aed',
        mountain: '#7c3aed',
        rock: '#1f2937',
        rockBorder: '#374151',
        dynamicRock: '#ef4444',
        base: '#2563eb',
        commZone: '#059669',
        dataSite: '#d97706',
        rover: '#00f2fe',
        roverGlow: 'rgba(0, 242, 254, 0.6)',
        unknown: '#090e1a',
        unknownGrid: 'rgba(255, 255, 255, 0.05)',
        unknownDot: 'rgba(255, 255, 255, 0.07)',
        visited: '#1d4ed8',
        visitedOverlay: 'rgba(29, 78, 216, 0.35)',
        visibleOverlay: 'rgba(56, 189, 248, 0.28)',
        frontier: '#d946ef',
        path: '#f43f5e',
        detectedObstacle: '#dc2626',
        gridLine: 'rgba(255, 255, 255, 0.06)',
    };

    // =========================================================================
    // API Communication
    // =========================================================================

    async function apiRequest(endpoint, method = 'GET', body = null) {
        try {
            const options = {
                method,
                headers: { 'Content-Type': 'application/json' },
            };
            if (body) {
                options.body = JSON.stringify(body);
            }
            const res = await fetch(endpoint, options);
            if (!res.ok) {
                console.error(`API Error ${res.status}: ${res.statusText}`);
                return null;
            }
            return await res.json();
        } catch (err) {
            console.error(`Network fetch failed for ${endpoint}:`, err);
            return null;
        }
    }

    async function fetchState() {
        const state = await apiRequest('/api/state');
        if (state) {
            updateDashboard(state);
        }
    }

    async function stepSimulation() {
        if (isStepping) return;
        isStepping = true;
        try {
            const state = await apiRequest('/api/step', 'POST');
            if (state) {
                updateDashboard(state);
                if (state.controls && state.controls.is_finished) {
                    stopLoop();
                }
            }
        } finally {
            isStepping = false;
        }
    }

    async function startSimulation() {
        if (latestState && latestState.controls && latestState.controls.is_finished) {
            return;
        }
        await apiRequest('/api/start', 'POST');
        isRunning = true;
        updateButtonStates();
        startLoop();
    }

    async function pauseSimulation() {
        await apiRequest('/api/pause', 'POST');
        isRunning = false;
        updateButtonStates();
        stopLoop();
    }

    async function resumeSimulation() {
        if (latestState && latestState.controls && latestState.controls.is_finished) {
            return;
        }
        await apiRequest('/api/resume', 'POST');
        isRunning = true;
        updateButtonStates();
        startLoop();
    }

    async function resetSimulation() {
        stopLoop();
        isRunning = false;
        const seedVal = parseInt(elements.inputSeed.value, 10) || 42;
        const dynObs = elements.toggleDynamicObs.checked;
        const state = await apiRequest('/api/reset', 'POST', {
            seed: seedVal,
            dynamic_obstacles: dynObs,
        });
        if (state) {
            updateDashboard(state);
        }
        updateButtonStates();
    }

    async function setSpeed(speedMs) {
        currentSpeedMs = speedMs;
        await apiRequest('/api/speed', 'POST', { speed_ms: speedMs });
        if (isRunning) {
            stopLoop();
            startLoop();
        }
    }

    function startLoop() {
        stopLoop();
        timerId = setInterval(async () => {
            if (isRunning) {
                await stepSimulation();
            }
        }, currentSpeedMs);
    }

    function stopLoop() {
        if (timerId !== null) {
            clearInterval(timerId);
            timerId = null;
        }
    }

    // =========================================================================
    // UI Updates & Telemetry
    // =========================================================================

    function updateButtonStates() {
        const isFinished = latestState && latestState.controls && latestState.controls.is_finished;
        elements.btnStart.disabled = isRunning || isFinished;
        elements.btnPause.disabled = !isRunning;
        elements.btnResume.disabled = isRunning || isFinished;
        elements.btnStep.disabled = isRunning || isFinished;
        elements.btnReset.disabled = false;
    }

    function updateDashboard(state) {
        latestState = state;
        const t = state.telemetry || {};
        const m = state.metrics || {};
        const d = state.dynamic_events || {};
        const c = state.controls || {};

        isRunning = c.is_running && !c.is_finished;
        updateButtonStates();

        // 1. Mission Status Badge
        updateStatusBadge(m.mission_outcome, t.is_stranded, t.is_at_base);

        // 2. Telemetry Tickers
        elements.stepCount.textContent = t.step_count ?? 0;
        elements.coverage.textContent = `${m.coverage_percent ?? 0.0}%`;
        elements.replans.textContent = m.replan_count ?? 0;

        // 3. Energy Subsystem
        const curEnergy = t.energy ?? 0.0;
        const maxEnergy = t.max_energy ?? 300.0;
        const energyPct = t.energy_percent ?? 100.0;
        elements.energyNumeric.textContent = `${curEnergy.toFixed(1)} / ${maxEnergy.toFixed(1)}`;
        elements.energyPercentage.textContent = `${energyPct.toFixed(1)}% Battery`;
        elements.energyUsedLabel.textContent = `Used: ${(m.energy_used ?? 0).toFixed(1)}`;
        elements.energyGaugeFill.style.width = `${Math.max(0, Math.min(100, energyPct))}%`;

        elements.energyGaugeFill.classList.remove('warning', 'danger');
        if (energyPct < 25) {
            elements.energyGaugeFill.classList.add('danger');
        } else if (energyPct < 50) {
            elements.energyGaugeFill.classList.add('warning');
        }

        // 4. Position & Safe Return
        if (t.position) {
            elements.roverPosDisplay.textContent = `(${t.position[0]}, ${t.position[1]})`;
        }
        elements.returnCostVal.textContent = t.return_cost;
        elements.safetyBufferVal.textContent = t.safety_buffer;
        elements.baseStatusTag.textContent = t.is_at_base ? 'At Base' : 'In Field';
        elements.baseStatusTag.className = t.is_at_base ? 'mini-value text-safe' : 'mini-value text-warning';

        // 5. Scientific Data
        elements.dataTotalYield.textContent = `${m.data_uploaded_value ?? 0} Value`;
        elements.dataCollectedVal.textContent = `${m.data_collected_count ?? 0} (${m.data_collected_value ?? 0})`;
        elements.dataUploadedVal.textContent = `${m.data_uploaded_count ?? 0} (${m.data_uploaded_value ?? 0})`;
        elements.dataUnuploadedVal.textContent = `${m.data_unuploaded_count ?? 0} (${m.data_unuploaded_value ?? 0})`;
        elements.dataLostVal.textContent = `${m.data_lost_count ?? 0} (${m.data_lost_value ?? 0})`;

        // 6. Directive / Goal
        if (t.current_goal) {
            const g = t.current_goal;
            elements.goalTypeBadge.textContent = (g.type || 'EXPLORE').toUpperCase();
            elements.goalTypeBadge.className = `badge-goal-${(g.type || 'explore').toLowerCase()}`;
            elements.goalTargetDesc.textContent = g.reason || `Target: (${g.target_pos[0]}, ${g.target_pos[1]})`;
            elements.goalUtilityVal.textContent = `Utility: ${g.utility_score}`;
            elements.goalCostVal.textContent = `Cost: ${g.estimated_cost}`;
        } else {
            elements.goalTypeBadge.textContent = t.is_stranded ? 'STRANDED' : 'IDLE';
            elements.goalTypeBadge.className = t.is_stranded ? 'badge-goal-return' : 'badge-goal-explore';
            elements.goalTargetDesc.textContent = t.last_action_reason || 'Standing by...';
            elements.goalUtilityVal.textContent = 'Utility: 0.0';
            elements.goalCostVal.textContent = 'Cost: 0.0';
        }

        // 7. Dynamic Obstacles & Known Cells Meta
        elements.gtDynamicCount.textContent = `Dynamic Obs: ${d.active_obstacles ? d.active_obstacles.length : 0}`;
        elements.beliefKnownCount.textContent = `Known: ${m.known_cells ?? 0} / ${m.total_cells ?? 1600}`;

        // 8. Terrain Cost Table
        updateTerrainCosts(t.estimated_terrain_costs, t.actual_terrain_costs);

        // 9. Logs
        renderLogs(state.logs || []);

        // 10. Canvases
        renderGroundTruthMap(state.ground_truth, t.position);
        renderBeliefMap(state.belief_map, t.position);

        // 11. Scientific Data Collection & Upload History
        renderDataHistory(state.data_history || [], state.data_history_summary || null);
    }

    function updateStatusBadge(outcome, isStranded, isAtBase) {
        const badge = elements.missionStatusBadge;
        badge.className = 'status-badge';

        if (isStranded || outcome === 'failure') {
            badge.textContent = 'STRANDED (FAILURE)';
            badge.classList.add('status-failure');
        } else if (outcome === 'success') {
            badge.textContent = 'MISSION SUCCESS';
            badge.classList.add('status-success');
        } else if (outcome === 'partial') {
            badge.textContent = 'PARTIAL SUCCESS';
            badge.classList.add('status-partial');
        } else {
            badge.textContent = 'IN PROGRESS';
            badge.classList.add('status-in-progress');
        }
    }

    function updateTerrainCosts(estCosts, actCosts) {
        if (!estCosts || !actCosts) return;

        function updateRow(estId, actId, accId, key) {
            const est = estCosts[key];
            const act = actCosts[key];
            const estEl = document.getElementById(estId);
            const actEl = document.getElementById(actId);
            const accEl = document.getElementById(accId);

            if (estEl && est !== undefined) estEl.textContent = est;
            if (actEl && act !== undefined) actEl.textContent = act;
            if (accEl && est !== undefined && act !== undefined) {
                if (typeof est === 'number' && typeof act === 'number') {
                    const acc = Math.max(0, 100 - Math.abs((est - act) / act) * 100);
                    accEl.textContent = `${acc.toFixed(0)}%`;
                    accEl.className = acc >= 90 ? 'text-safe' : 'text-warning';
                } else {
                    accEl.textContent = 'Exact';
                    accEl.className = 'text-safe';
                }
            }
        }

        updateRow('cost-flat-est', 'cost-flat-act', 'cost-flat-acc', 'flat');
        updateRow('cost-rough-est', 'cost-rough-act', 'cost-rough-acc', 'rough');
        const hillKey = estCosts['hill'] !== undefined ? 'hill' : 'mountain';
        updateRow('cost-mountain-est', 'cost-mountain-act', 'cost-mountain-acc', hillKey);
    }

    function renderLogs(logs) {
        elements.logCountBadge.textContent = `${logs.length} Events`;
        elements.decisionLogContainer.innerHTML = '';

        for (let i = logs.length - 1; i >= 0; i--) {
            const entry = logs[i];
            const div = document.createElement('div');
            const lvl = entry.level || 'info';
            div.className = `log-entry log-${lvl}`;

            const badgeClass = `badge-${lvl === 'danger' ? 'danger' : lvl === 'warning' ? 'warning' : lvl === 'success' ? 'success' : 'info'}`;

            div.innerHTML = `
                <span class="log-step">[STEP ${entry.step}]</span>
                <span class="log-badge ${badgeClass}">${lvl.toUpperCase()}</span>
                <span class="log-msg">${escapeHtml(entry.message)}</span>
            `;
            elements.decisionLogContainer.appendChild(div);
        }
    }

    function renderDataHistory(history, summary) {
        if (!elements.dataHistoryTbody) return;

        // 1. Update summary pills
        if (summary) {
            if (elements.pillValCollected) {
                elements.pillValCollected.textContent = `Collected: ${summary.total_collected_value ?? 0} Val (${summary.collected_count ?? 0})`;
            }
            if (elements.pillValUploaded) {
                elements.pillValUploaded.textContent = `Uploaded: ${summary.total_uploaded_value ?? 0} Val (${summary.uploaded_count ?? 0})`;
            }
            if (elements.pillValPending) {
                elements.pillValPending.textContent = `Pending: ${summary.total_pending_value ?? 0} Val (${summary.pending_count ?? 0})`;
            }
            if (elements.pillValLost) {
                elements.pillValLost.textContent = `Lost: ${summary.total_lost_value ?? 0} Val (${summary.lost_count ?? 0})`;
            }
        } else if (history) {
            const collectedVal = history.reduce((acc, r) => acc + (r.value || 0), 0);
            const uploadedVal = history.filter(r => r.upload_status === 'Uploaded').reduce((acc, r) => acc + (r.value || 0), 0);
            const pendingVal = history.filter(r => r.upload_status === 'Pending Upload').reduce((acc, r) => acc + (r.value || 0), 0);
            const lostVal = history.filter(r => r.upload_status === 'Lost').reduce((acc, r) => acc + (r.value || 0), 0);
            if (elements.pillValCollected) elements.pillValCollected.textContent = `Collected: ${collectedVal} Val`;
            if (elements.pillValUploaded) elements.pillValUploaded.textContent = `Uploaded: ${uploadedVal} Val`;
            if (elements.pillValPending) elements.pillValPending.textContent = `Pending: ${pendingVal} Val`;
            if (elements.pillValLost) elements.pillValLost.textContent = `Lost: ${lostVal} Val`;
        }

        // 2. Empty state check
        if (!history || history.length === 0) {
            elements.dataHistoryTbody.innerHTML = '';
            if (elements.dataHistoryEmptyMsg) {
                elements.dataHistoryEmptyMsg.style.display = 'flex';
            }
            return;
        }

        if (elements.dataHistoryEmptyMsg) {
            elements.dataHistoryEmptyMsg.style.display = 'none';
        }

        // 3. Render table rows
        let html = '';
        for (const record of history) {
            const uploadStatus = record.upload_status || 'Pending Upload';
            let uploadStatusClass = 'badge-status-pending';
            if (uploadStatus === 'Uploaded') {
                uploadStatusClass = 'badge-status-uploaded';
            } else if (uploadStatus === 'Lost') {
                uploadStatusClass = 'badge-status-lost';
            }

            // Data Site Coordinates (row, column)
            const rowCoord = record.site_row !== undefined ? record.site_row : record.site_y;
            const colCoord = record.site_col !== undefined ? record.site_col : record.site_x;
            const coordsStr = `(${rowCoord}, ${colCoord})`;

            // Communication Zone and Upload Step
            let uploadZoneDisplay = '—';
            let uploadStepDisplay = '—';
            if (uploadStatus === 'Uploaded') {
                uploadZoneDisplay = escapeHtml(record.upload_comm_zone || record.upload_comm_zone_id || 'Comm Zone');
                uploadStepDisplay = record.upload_step !== undefined && record.upload_step !== null && record.upload_step !== '—'
                    ? `Step ${record.upload_step}`
                    : '—';
            } else if (uploadStatus === 'Pending Upload') {
                uploadZoneDisplay = '<span class="status-cell-pending">Pending Upload</span>';
                uploadStepDisplay = '—';
            } else if (uploadStatus === 'Lost') {
                uploadZoneDisplay = '<span class="text-muted">—</span>';
                uploadStepDisplay = '—';
            }

            const dataIdDisplay = record.display_id || record.data_name || `DS-${record.data_id}`;

            html += `
                <tr>
                    <td class="font-mono text-accent" style="font-weight:700;">${escapeHtml(dataIdDisplay)}</td>
                    <td class="font-mono">${coordsStr}</td>
                    <td class="font-mono text-warning" style="font-weight:700;">+${record.value}</td>
                    <td class="font-mono">Step ${record.collection_step}</td>
                    <td><span class="history-badge badge-status-collected">${escapeHtml(record.collection_status || 'Collected')}</span></td>
                    <td class="font-mono">${uploadZoneDisplay}</td>
                    <td class="font-mono">${uploadStepDisplay}</td>
                    <td><span class="history-badge ${uploadStatusClass}">${escapeHtml(uploadStatus)}</span></td>
                </tr>
            `;
        }
        elements.dataHistoryTbody.innerHTML = html;
    }

    async function fetchExperimentResults() {
        if (!elements.strategyEvalTbody) return;
        try {
            const data = await apiRequest('/api/experiments', 'GET');
            if (!data || !data.available || !data.summary) {
                if (elements.evalEmptyMsg) elements.evalEmptyMsg.style.display = 'flex';
                if (elements.evalBenchmarkInfo) elements.evalBenchmarkInfo.textContent = 'Benchmark Pending';
                return;
            }

            if (elements.evalEmptyMsg) elements.evalEmptyMsg.style.display = 'none';
            const s = data.summary;
            if (elements.evalBenchmarkInfo) {
                elements.evalBenchmarkInfo.textContent = `${s.total_seeds} Seeds · ${s.total_runs} Runs Benchmark`;
            }

            const strats = s.strategies || {};
            const order = ['random', 'greedy_nearest', 'utility_planner'];
            const sortedKeys = Object.keys(strats).sort((a, b) => {
                const ia = order.indexOf(a) !== -1 ? order.indexOf(a) : 99;
                const ib = order.indexOf(b) !== -1 ? order.indexOf(b) : 99;
                return ia - ib;
            });

            let html = '';
            for (const key of sortedKeys) {
                const item = strats[key];
                const isProposed = key === 'utility_planner';
                const rowClass = isProposed ? 'strat-row-highlight' : '';
                const icon = key === 'random' ? '🎲' : key === 'greedy_nearest' ? '⚡' : '🧠';
                const tag = isProposed ? '<span class="badge-tag-proposed">PROPOSED</span>' : '';

                html += `
                    <tr class="${rowClass}">
                        <td>
                            <div class="strategy-title-cell">
                                <span class="strat-icon">${icon}</span>
                                <span>${escapeHtml(item.display_name)}</span>
                                ${tag}
                            </div>
                        </td>
                        <td><span class="status-badge status-success">${item.success_rate_percent}%</span> (${item.success_count})</td>
                        <td><span class="status-badge status-partial">${item.partial_rate_percent}%</span> (${item.partial_count})</td>
                        <td><span class="status-badge status-failure">${item.failure_rate_percent}%</span> (${item.failure_count})</td>
                        <td class="text-accent">${item.avg_coverage_percent}%</td>
                        <td class="text-safe">${item.avg_data_uploaded_value}</td>
                        <td class="${item.avg_data_lost_value > 0 ? 'text-danger' : 'text-muted'}">${item.avg_data_lost_value}</td>
                        <td>${item.avg_energy_used}</td>
                        <td>${item.avg_replan_count}</td>
                    </tr>
                `;
            }
            elements.strategyEvalTbody.innerHTML = html;
        } catch (err) {
            console.error('Failed to fetch experiment results:', err);
        }
    }

    function escapeHtml(str) {
        return (str || '').replace(/[&<>"']/g, function (m) {
            return {
                '&': '&amp;',
                '<': '&lt;',
                '>': '&gt;',
                '"': '&quot;',
                "'": '&#39;',
            }[m];
        });
    }

    // =========================================================================
    // Canvas Map Renderers
    // =========================================================================

    function renderGroundTruthMap(gt, roverPos) {
        if (!gt || !gt.grid) return;
        const width = elements.canvasGT.width;
        const height = elements.canvasGT.height;
        const cols = gt.width || 40;
        const rows = gt.height || 40;
        const cellW = width / cols;
        const cellH = height / rows;

        ctxGT.fillStyle = COLORS.bg;
        ctxGT.fillRect(0, 0, width, height);

        // Draw Cells
        for (let y = 0; y < rows; y++) {
            for (let x = 0; x < cols; x++) {
                const c = gt.grid[y][x];
                let color = COLORS.flat;

                if (c.terrain === 'rough') color = COLORS.rough;
                else if (c.terrain === 'hill' || c.terrain === 'mountain') color = COLORS.mountain;
                else if (c.terrain === 'rock' || c.is_obstacle) color = COLORS.rock;

                if (c.is_comm_zone) color = COLORS.commZone;
                if (c.is_base) color = COLORS.base;
                if (c.data_site) color = COLORS.dataSite;
                if (c.is_dynamic_obstacle) color = COLORS.dynamicRock;

                ctxGT.fillStyle = color;
                ctxGT.fillRect(x * cellW, y * cellH, cellW, cellH);

                // Grid line border
                ctxGT.strokeStyle = COLORS.gridLine;
                ctxGT.strokeRect(x * cellW, y * cellH, cellW, cellH);

                // Highlight special nodes
                if (c.is_base) {
                    ctxGT.fillStyle = '#ffffff';
                    ctxGT.font = `bold ${Math.max(8, cellW * 0.7)}px sans-serif`;
                    ctxGT.textAlign = 'center';
                    ctxGT.textBaseline = 'middle';
                    ctxGT.fillText('B', x * cellW + cellW / 2, y * cellH + cellH / 2);
                } else if (c.data_site) {
                    ctxGT.fillStyle = '#ffffff';
                    ctxGT.font = `bold ${Math.max(8, cellW * 0.65)}px sans-serif`;
                    ctxGT.textAlign = 'center';
                    ctxGT.textBaseline = 'middle';
                    ctxGT.fillText('D', x * cellW + cellW / 2, y * cellH + cellH / 2);
                } else if (c.is_comm_zone) {
                    ctxGT.fillStyle = '#ffffff';
                    ctxGT.font = `bold ${Math.max(8, cellW * 0.65)}px sans-serif`;
                    ctxGT.textAlign = 'center';
                    ctxGT.textBaseline = 'middle';
                    ctxGT.fillText('C', x * cellW + cellW / 2, y * cellH + cellH / 2);
                } else if (c.is_dynamic_obstacle) {
                    ctxGT.fillStyle = '#ffffff';
                    ctxGT.font = `bold ${Math.max(8, cellW * 0.7)}px sans-serif`;
                    ctxGT.textAlign = 'center';
                    ctxGT.textBaseline = 'middle';
                    ctxGT.fillText('!', x * cellW + cellW / 2, y * cellH + cellH / 2);
                }
            }
        }

        // Draw Rover Marker
        if (roverPos) {
            const rx = roverPos[0] * cellW + cellW / 2;
            const ry = roverPos[1] * cellH + cellH / 2;
            const radius = cellW * 0.7;

            // Halo glow
            ctxGT.beginPath();
            ctxGT.arc(rx, ry, radius * 1.5, 0, Math.PI * 2);
            ctxGT.fillStyle = COLORS.roverGlow;
            ctxGT.fill();

            // Rover core
            ctxGT.beginPath();
            ctxGT.arc(rx, ry, radius, 0, Math.PI * 2);
            ctxGT.fillStyle = COLORS.rover;
            ctxGT.fill();

            ctxGT.fillStyle = '#050b14';
            ctxGT.font = `bold ${Math.max(8, cellW * 0.75)}px sans-serif`;
            ctxGT.textAlign = 'center';
            ctxGT.textBaseline = 'middle';
            ctxGT.fillText('R', rx, ry);
        }
    }

    function renderBeliefMap(belief, roverPos) {
        if (!belief || !belief.grid) return;
        const width = elements.canvasBelief.width;
        const height = elements.canvasBelief.height;
        const cols = belief.width || 40;
        const rows = belief.height || 40;
        const cellW = width / cols;
        const cellH = height / rows;

        ctxBelief.fillStyle = COLORS.bg;
        ctxBelief.fillRect(0, 0, width, height);

        // Draw Cells
        for (let y = 0; y < rows; y++) {
            for (let x = 0; x < cols; x++) {
                const c = belief.grid[y][x];

                if (c.is_known) {
                    let color = COLORS.flat;
                    if (c.terrain === 'rough') color = COLORS.rough;
                    else if (c.terrain === 'hill' || c.terrain === 'mountain') color = COLORS.mountain;
                    else if (c.terrain === 'rock' || c.is_obstacle || c.is_detected_obstacle) color = COLORS.detectedObstacle;
                    else color = COLORS.flat;

                    if (c.is_visited) color = COLORS.visited;
                    if (c.is_comm_zone) color = COLORS.commZone;
                    if (c.is_base) color = COLORS.base;
                    if (c.data_site) color = COLORS.dataSite;
                    if (c.is_obstacle || c.is_detected_obstacle) color = COLORS.detectedObstacle;

                    ctxBelief.fillStyle = color;
                    ctxBelief.fillRect(x * cellW, y * cellH, cellW, cellH);

                    // Grid lines for known cells
                    ctxBelief.strokeStyle = COLORS.gridLine;
                    ctxBelief.strokeRect(x * cellW, y * cellH, cellW, cellH);
                } else {
                    // Unknown cell: Distinct cosmic fog of war with uncharted grid and center dot
                    ctxBelief.fillStyle = COLORS.unknown;
                    ctxBelief.fillRect(x * cellW, y * cellH, cellW, cellH);

                    ctxBelief.strokeStyle = COLORS.unknownGrid;
                    ctxBelief.strokeRect(x * cellW, y * cellH, cellW, cellH);

                    ctxBelief.fillStyle = COLORS.unknownDot;
                    ctxBelief.fillRect(
                        x * cellW + cellW * 0.4,
                        y * cellH + cellH * 0.4,
                        Math.max(1, cellW * 0.2),
                        Math.max(1, cellH * 0.2)
                    );
                }

                // Radar Sensory Field (Currently Visible Cells)
                if (c.is_visible) {
                    ctxBelief.fillStyle = COLORS.visibleOverlay;
                    ctxBelief.fillRect(x * cellW, y * cellH, cellW, cellH);
                }

                // Frontier indicator
                if (c.is_frontier) {
                    ctxBelief.strokeStyle = COLORS.frontier;
                    ctxBelief.lineWidth = 1.5;
                    ctxBelief.strokeRect(x * cellW + 1, y * cellH + 1, cellW - 2, cellH - 2);
                    ctxBelief.lineWidth = 1.0;
                    ctxBelief.fillStyle = COLORS.frontier;
                    ctxBelief.fillRect(x * cellW + cellW * 0.35, y * cellH + cellH * 0.35, cellW * 0.3, cellH * 0.3);
                }

                // Sensed Special Markers
                if (c.is_known) {
                    if (c.is_base) {
                        ctxBelief.fillStyle = '#ffffff';
                        ctxBelief.font = `bold ${Math.max(8, cellW * 0.7)}px sans-serif`;
                        ctxBelief.textAlign = 'center';
                        ctxBelief.textBaseline = 'middle';
                        ctxBelief.fillText('B', x * cellW + cellW / 2, y * cellH + cellH / 2);
                    } else if (c.data_site) {
                        ctxBelief.fillStyle = '#ffffff';
                        ctxBelief.font = `bold ${Math.max(8, cellW * 0.65)}px sans-serif`;
                        ctxBelief.textAlign = 'center';
                        ctxBelief.textBaseline = 'middle';
                        ctxBelief.fillText('D', x * cellW + cellW / 2, y * cellH + cellH / 2);
                    } else if (c.is_comm_zone) {
                        ctxBelief.fillStyle = '#ffffff';
                        ctxBelief.font = `bold ${Math.max(8, cellW * 0.65)}px sans-serif`;
                        ctxBelief.textAlign = 'center';
                        ctxBelief.textBaseline = 'middle';
                        ctxBelief.fillText('C', x * cellW + cellW / 2, y * cellH + cellH / 2);
                    } else if (c.is_obstacle || c.is_detected_obstacle) {
                        ctxBelief.fillStyle = '#ffffff';
                        ctxBelief.font = `bold ${Math.max(8, cellW * 0.7)}px sans-serif`;
                        ctxBelief.textAlign = 'center';
                        ctxBelief.textBaseline = 'middle';
                        ctxBelief.fillText('X', x * cellW + cellW / 2, y * cellH + cellH / 2);
                    }
                }
            }
        }

        // Draw Active Route Path Waypoints
        if (latestState && latestState.telemetry && latestState.telemetry.current_path) {
            const path = latestState.telemetry.current_path;
            if (path && path.length > 1) {
                ctxBelief.strokeStyle = COLORS.path;
                ctxBelief.lineWidth = Math.max(2, cellW * 0.3);
                ctxBelief.lineJoin = 'round';
                ctxBelief.beginPath();
                for (let i = 0; i < path.length; i++) {
                    const px = path[i][0] * cellW + cellW / 2;
                    const py = path[i][1] * cellH + cellH / 2;
                    if (i === 0) ctxBelief.moveTo(px, py);
                    else ctxBelief.lineTo(px, py);
                }
                ctxBelief.stroke();

                // Draw path waypoint dots
                ctxBelief.fillStyle = '#ffffff';
                for (let i = 1; i < path.length; i++) {
                    const px = path[i][0] * cellW + cellW / 2;
                    const py = path[i][1] * cellH + cellH / 2;
                    ctxBelief.beginPath();
                    ctxBelief.arc(px, py, cellW * 0.2, 0, Math.PI * 2);
                    ctxBelief.fill();
                }
            }
        }

        // Draw Rover Marker
        if (roverPos) {
            const rx = roverPos[0] * cellW + cellW / 2;
            const ry = roverPos[1] * cellH + cellH / 2;
            const radius = cellW * 0.7;

            // Halo glow
            ctxBelief.beginPath();
            ctxBelief.arc(rx, ry, radius * 1.5, 0, Math.PI * 2);
            ctxBelief.fillStyle = COLORS.roverGlow;
            ctxBelief.fill();

            // Rover core
            ctxBelief.beginPath();
            ctxBelief.arc(rx, ry, radius, 0, Math.PI * 2);
            ctxBelief.fillStyle = COLORS.rover;
            ctxBelief.fill();

            ctxBelief.fillStyle = '#050b14';
            ctxBelief.font = `bold ${Math.max(8, cellW * 0.75)}px sans-serif`;
            ctxBelief.textAlign = 'center';
            ctxBelief.textBaseline = 'middle';
            ctxBelief.fillText('R', rx, ry);
        }
    }

    // =========================================================================
    // Interactive Tooltips
    // =========================================================================

    function setupCanvasTooltips() {
        function handleMouseMove(e, canvas, tooltip, isBelief) {
            if (!latestState) return;
            const mapData = isBelief ? latestState.belief_map : latestState.ground_truth;
            if (!mapData || !mapData.grid) return;

            const rect = canvas.getBoundingClientRect();
            const mouseX = e.clientX - rect.left;
            const mouseY = e.clientY - rect.top;

            const cols = mapData.width || 40;
            const rows = mapData.height || 40;
            const cellW = rect.width / cols;
            const cellH = rect.height / rows;

            const cellX = Math.floor(mouseX / cellW);
            const cellY = Math.floor(mouseY / cellH);

            if (cellX < 0 || cellX >= cols || cellY < 0 || cellY >= rows) {
                tooltip.style.display = 'none';
                return;
            }

            const cell = mapData.grid[cellY][cellX];
            let content = `<strong>Coord:</strong> (${cellX}, ${cellY})<br>`;

            if (isBelief) {
                if (cell.is_unknown) {
                    content += `<strong>Status:</strong> UNKNOWN (?)`;
                } else {
                    content += `<strong>Terrain:</strong> ${cell.terrain || 'Observed'}<br>`;
                    content += `<strong>Cost:</strong> ${cell.cost}<br>`;
                    if (cell.experienced_cost) content += `<strong>Experienced:</strong> ${cell.experienced_cost.toFixed(1)}<br>`;
                    if (cell.is_base) content += `★ Base Station<br>`;
                    if (cell.is_comm_zone) content += `★ Comm Zone<br>`;
                    if (cell.data_site) content += `★ Data Site (Val: ${cell.data_site.value})<br>`;
                    if (cell.is_detected_obstacle) content += `⚠ Detected Obstacle [X]<br>`;
                    if (cell.is_frontier) content += `◈ Exploration Frontier<br>`;
                    if (cell.is_visible) content += `◉ Within Sensor Cone<br>`;
                    if (cell.is_rover) content += `🚀 ROVER POSITION<br>`;
                }
            } else {
                content += `<strong>Terrain:</strong> ${cell.terrain}<br>`;
                content += `<strong>Cost:</strong> ${cell.cost}<br>`;
                if (cell.is_base) content += `★ Base Station<br>`;
                if (cell.is_comm_zone) content += `★ Comm Zone<br>`;
                if (cell.data_site) content += `★ Data Site: ${cell.data_site.name} (Val: ${cell.data_site.value})<br>`;
                if (cell.is_dynamic_obstacle) content += `⚠ Dynamic Obstacle [!]<br>`;
                else if (cell.is_obstacle) content += `Static Rock Obstacle<br>`;
                if (cell.is_rover) content += `🚀 ROVER POSITION<br>`;
            }

            tooltip.innerHTML = content;
            tooltip.style.display = 'block';

            // Position tooltip safely inside canvas boundary
            const pad = 12;
            let left = mouseX + pad;
            let top = mouseY + pad;
            if (left + 180 > rect.width) left = mouseX - 190;
            if (top + 100 > rect.height) top = mouseY - 90;

            tooltip.style.left = `${left}px`;
            tooltip.style.top = `${top}px`;
        }

        elements.canvasGT.addEventListener('mousemove', (e) => handleMouseMove(e, elements.canvasGT, elements.tooltipGT, false));
        elements.canvasGT.addEventListener('mouseleave', () => { elements.tooltipGT.style.display = 'none'; });

        elements.canvasBelief.addEventListener('mousemove', (e) => handleMouseMove(e, elements.canvasBelief, elements.tooltipBelief, true));
        elements.canvasBelief.addEventListener('mouseleave', () => { elements.tooltipBelief.style.display = 'none'; });
    }

    // =========================================================================
    // Event Listeners & Initialization
    // =========================================================================

    function setupEventListeners() {
        elements.btnStart.addEventListener('click', startSimulation);
        elements.btnPause.addEventListener('click', pauseSimulation);
        elements.btnResume.addEventListener('click', resumeSimulation);
        elements.btnStep.addEventListener('click', stepSimulation);
        elements.btnReset.addEventListener('click', resetSimulation);

        elements.selectSpeed.addEventListener('change', (e) => {
            setSpeed(parseInt(e.target.value, 10));
        });

        // Enter key in seed input triggers reset
        elements.inputSeed.addEventListener('keydown', (e) => {
            if (e.key === 'Enter') {
                resetSimulation();
            }
        });

        if (elements.btnRefreshEval) {
            elements.btnRefreshEval.addEventListener('click', fetchExperimentResults);
        }
    }

    // Initialize Dashboard
    setupEventListeners();
    setupCanvasTooltips();
    fetchState();
    fetchExperimentResults();

})();
