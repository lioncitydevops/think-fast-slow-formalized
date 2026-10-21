document.addEventListener("DOMContentLoaded", () => {
  // Navigation Tabs
  const tabBtns = document.querySelectorAll(".tab-btn");
  const tabPanes = document.querySelectorAll(".tab-pane");

  tabBtns.forEach(btn => {
    btn.addEventListener("click", () => {
      tabBtns.forEach(b => b.classList.remove("active"));
      tabPanes.forEach(p => p.classList.remove("active"));
      btn.classList.add("active");
      const target = btn.dataset.tab;
      document.getElementById(target).classList.add("active");
    });
  });

  // Pattern Chips
  const patternChips = document.querySelectorAll(".pattern-chip");
  let selectedPattern = "P3";

  patternChips.forEach(chip => {
    chip.addEventListener("click", () => {
      patternChips.forEach(c => c.classList.remove("active"));
      chip.classList.add("active");
      selectedPattern = chip.dataset.pattern;
      updatePatternHelp(selectedPattern);
    });
  });

  function updatePatternHelp(p) {
    const helpEl = document.getElementById("patternHelp");
    const desc = {
      P1: "P1 Cascade: Default-interventionist. Fast Jev check; deliberates with fixed budget if c < tau.",
      P2: "P2 Budgeted: Metacognitive allocation. Deliberates with task-conditioned budget B*(c, task).",
      P3: "P3 Disagreement Probe: Parallel-competitive. Runs fast probe a0; deliberates only on conflict.",
      P4: "P4 Jev as Verifier: When Qwen overturns Jev, Jev audits whether reasoning is factually supported.",
      P5: "P5 Blind-then-Reconcile: Qwen reasons blind, then audits reasoning seeing (a1, c) in a second turn.",
    };
    if (helpEl) helpEl.textContent = desc[p] || "";
  }

  // Load Status
  fetch("/api/status")
    .then(r => r.json())
    .then(status => {
      const s1Dot = document.getElementById("s1Dot");
      const s1Text = document.getElementById("s1Text");
      const s2Dot = document.getElementById("s2Dot");
      const s2Text = document.getElementById("s2Text");

      if (status.system1.configured) {
        s1Dot.className = "status-dot";
        s1Text.textContent = `Jev: Online (${status.system1.model})`;
      } else {
        s1Dot.className = "status-dot secondary";
        s1Text.textContent = "Jev: Offline / Mock";
      }

      if (status.system2.online) {
        s2Dot.className = "status-dot";
        s2Text.textContent = "Qwen3-8B: Online (RTX 3070 GPU)";
      } else {
        s2Dot.className = "status-dot secondary";
        s2Text.textContent = "Qwen3-8B: Mock Engine";
      }
    })
    .catch(() => {});

  // Load Domains
  let domainsData = {};
  fetch("/api/domains")
    .then(r => r.json())
    .then(data => {
      domainsData = data;
      renderDomainsGrid(data);
    });

  function renderDomainsGrid(domains) {
    const container = document.getElementById("domainsGrid");
    if (!container) return;
    container.innerHTML = "";

    Object.entries(domains).forEach(([key, d]) => {
      const card = document.createElement("div");
      card.className = "domain-card";
      card.innerHTML = `
        <span class="domain-tag">${key.toUpperCase()}</span>
        <h3>${d.name}</h3>
        <p style="font-size: 13px; color: var(--text-muted); margin-bottom: 12px;">${d.system2_task}</p>
        <div style="font-size: 12px; color: var(--accent-rose); margin-bottom: 8px;"><b>Costly Error:</b> ${d.costly_error}</div>
        <div style="font-size: 12px; color: var(--accent-emerald);"><b>Fallback:</b> ${d.fallback_action}</div>
      `;
      card.addEventListener("click", () => {
        loadPresetQuery(d.sample_queries[0], key);
        // Switch to playground tab
        document.querySelector('[data-tab="playgroundTab"]').click();
      });
      container.appendChild(card);
    });
  }

  function loadPresetQuery(q, domainKey) {
    if (!q) return;
    document.getElementById("queryText").value = q.text;
    document.getElementById("queryQuestion").value = q.question;
    document.getElementById("queryOptions").value = q.options.join(", ");
    document.getElementById("taskDomain").value = domainKey;
  }

  // Load Replications Data (Table 3 & Table 5)
  fetch("/api/replications")
    .then(r => r.json())
    .then(data => {
      renderTable3(data.table_3);
      renderTable5(data.table_5);
    });

  function renderTable3(rows) {
    const tbody = document.querySelector("#table3Body");
    if (!tbody || !rows) return;
    tbody.innerHTML = "";
    rows.forEach(r => {
      const tr = document.createElement("tr");
      tr.innerHTML = `
        <td style="font-weight: 600; color: #fff;">${r.curve}</td>
        <td style="font-family: var(--font-mono); color: var(--accent-emerald);">${r.act_c.toFixed(2)}</td>
        <td style="font-family: var(--font-mono); color: var(--accent-rose);">${r.give_up_c.toFixed(2)}</td>
        <td style="font-family: var(--font-mono);">${r.peak_B.toLocaleString()}</td>
        <td style="font-family: var(--font-mono);">${r.min_B_pos.toLocaleString()}</td>
        <td style="font-family: var(--font-mono); color: var(--accent-cyan);">${Math.round(r.kept * 100)}%</td>
      `;
      tbody.appendChild(tr);
    });
  }

  function renderTable5(rows) {
    const tbody = document.querySelector("#table5Body");
    if (!tbody || !rows) return;
    tbody.innerHTML = "";
    rows.forEach(r => {
      const tr = document.createElement("tr");
      tr.innerHTML = `
        <td style="font-weight: 600; color: #fff;">${r.policy}</td>
        <td style="font-family: var(--font-mono);">${r.act_share}</td>
        <td style="font-family: var(--font-mono);">${r.deliberate_share}</td>
        <td style="font-family: var(--font-mono);">${r.give_up_share}</td>
        <td style="font-family: var(--font-mono); color: var(--accent-cyan);">${r.tokens}</td>
        <td style="font-family: var(--font-mono); color: var(--accent-emerald); font-weight: 700;">${r.accuracy.toFixed(3)}</td>
      `;
      tbody.appendChild(tr);
    });
  }

  // Execute Query Action
  const runBtn = document.getElementById("runQueryBtn");
  runBtn.addEventListener("click", () => {
    const text = document.getElementById("queryText").value.trim();
    const question = document.getElementById("queryQuestion").value.trim();
    const optionsRaw = document.getElementById("queryOptions").value.trim();
    const tau = parseFloat(document.getElementById("tauThreshold").value);
    const useMock = document.getElementById("useMockToggle").checked;
    const task = document.getElementById("taskDomain").value;

    if (!text || !question || !optionsRaw) {
      alert("Please fill in Text State, Question, and Options.");
      return;
    }

    const options = optionsRaw.split(",").map(o => o.trim()).filter(Boolean);

    runBtn.disabled = true;
    runBtn.innerHTML = `<span class="status-dot"></span> Processing Cognitive Pipeline...`;

    fetch("/api/query", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        text,
        question,
        options,
        pattern: selectedPattern,
        tau,
        use_mock: useMock,
        task,
        probe: selectedPattern === "P3",
        reconcile: selectedPattern === "P5",
        verify: selectedPattern === "P4",
      }),
    })
      .then(r => r.json())
      .then(res => {
        renderPipelineResults(res);
      })
      .catch(err => {
        alert("Error executing query: " + err.message);
      })
      .finally(() => {
        runBtn.disabled = false;
        runBtn.innerHTML = `⚡ Execute Dual-Process Decision`;
      });
  });

  function renderPipelineResults(res) {
    // Stage 1 Jev
    const s1Choice = document.getElementById("s1Choice");
    const s1Conf = document.getElementById("s1Conf");
    const s1Latency = document.getElementById("s1Latency");
    const s1ProbList = document.getElementById("s1ProbList");

    s1Choice.textContent = res.system1.action;
    s1Conf.textContent = `${Math.round(res.system1.confidence * 100)}%`;
    s1Latency.textContent = `${res.system1.latency.toFixed(3)}s`;

    s1ProbList.innerHTML = "";
    Object.entries(res.system1.probabilities).forEach(([opt, p]) => {
      const pct = Math.round(p * 100);
      const row = document.createElement("div");
      row.className = "prob-row";
      row.innerHTML = `
        <div class="prob-label" title="${opt}">${opt}</div>
        <div class="prob-bar-bg">
          <div class="prob-bar-fill" style="width: ${pct}%;"></div>
        </div>
        <div class="prob-pct">${pct}%</div>
      `;
      s1ProbList.appendChild(row);
    });

    // Stage 2 Metacognitive Control
    const bandBadge = document.getElementById("decisionBandBadge");
    bandBadge.textContent = res.band;
    bandBadge.className = "band-indicator";
    if (res.band === "ACT") bandBadge.classList.add("band-act");
    else if (res.band === "DELIBERATE") bandBadge.classList.add("band-deliberate");
    else bandBadge.classList.add("band-give-up");

    document.getElementById("execMode").textContent = res.mode;
    document.getElementById("tokensSpent").textContent = res.tokens_spent;
    document.getElementById("totalLatency").textContent = `${res.total_latency.toFixed(2)}s`;

    // Stage 3 Qwen Deliberation / Fallback
    const s2Choice = document.getElementById("s2Choice");
    const s2Trace = document.getElementById("s2Trace");
    const conformalEl = document.getElementById("conformalSet");

    if (res.system2) {
      s2Choice.textContent = res.system2.action;
      s2Trace.textContent = res.system2.reasoning_trace || "[No deliberation needed - Law of Least Effort]";
    } else {
      s2Choice.textContent = "[Not invoked - Heuristic Act]";
      s2Trace.textContent = `Law of Least Effort: Jev confidence (${res.system1.confidence}) >= tau (${document.getElementById("tauThreshold").value}). Immediate heuristic release.`;
    }

    if (res.conformal_set) {
      conformalEl.textContent = `{ ${res.conformal_set.join(", ")} }`;
    }

    // Final Action Box
    document.getElementById("finalReleasedAction").textContent = res.action;
  }

  // Run Evaluation Suite Action
  const evalBtn = document.getElementById("runEvalBtn");
  if (evalBtn) {
    evalBtn.addEventListener("click", () => {
      evalBtn.disabled = true;
      evalBtn.textContent = "Running Pre-Registered Harness...";

      fetch("/api/evaluate?samples=20", { method: "POST" })
        .then(r => r.json())
        .then(res => {
          renderEvaluationResults(res);
        })
        .finally(() => {
          evalBtn.disabled = false;
          evalBtn.textContent = "Run Full Evaluation Suite";
        });
    });
  }

  function renderEvaluationResults(res) {
    const listEl = document.getElementById("hypothesesList");
    if (!listEl) return;
    listEl.innerHTML = "";

    res.hypotheses.forEach(h => {
      const card = document.createElement("div");
      card.className = "glass-card";
      card.style.padding = "16px";
      card.style.marginBottom = "12px";

      const isPass = !h.falsified;
      const badgeClass = isPass ? "badge-pass" : "badge-fail";
      const statusText = isPass ? "VERIFIED (SUPPORTED)" : "FALSIFIED";

      card.innerHTML = `
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
          <b style="color: #fff; font-size: 15px;">${h.id}: ${h.description}</b>
          <span class="${badgeClass}">${statusText}</span>
        </div>
        <div style="font-size: 12px; color: var(--text-muted); margin-bottom: 4px;">Falsification criterion: <i>${h.falsification_criteria}</i></div>
        <div style="font-size: 12px; font-family: var(--font-mono); color: var(--accent-cyan);">${h.metric_value}</div>
      `;
      listEl.appendChild(card);
    });

    // Escalation identity check
    const esc = res.escalation_identity;
    document.getElementById("escIdentityDiff").textContent = `${esc.diff_machine_precision} (Precision Holds: ${esc.identity_holds})`;
    document.getElementById("escMZ").textContent = esc.M_Z.toFixed(3);
    document.getElementById("escWinMargin").textContent = (esc.win_margin || 0).toFixed(3);

    // Paired decomposition check
    const decomp = res.paired_decomposition;
    document.getElementById("decompDiff").textContent = `${decomp.diff_machine_precision} (Precision Holds: ${decomp.identity_holds})`;
    document.getElementById("decompStableZero").textContent = `${decomp.stable_contribution_zero}`;
    document.getElementById("decompAnchorShare").textContent = `${(decomp.shares.anchor * 100).toFixed(1)}%`;
  }
});
