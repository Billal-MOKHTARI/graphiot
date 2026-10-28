# Progress log

## 2026-10-24 — planning + M1 core (physical testbed) done and tested

**Decisions locked** (see PLAN.md §6): OpenPLC Runtime v4; local Ollama (operator
llama3.1:8b / red-team qwen2.5 7B); caches cleared; oracle = primary twin replay with
τ=0.05, T_min=10 s, my proposed limits (config-overridable for a real OSF registration).

**Built and tested (22 pytest tests green, 0.67 s):**
- `src/graphiot/plant/tags.py` + `plant/tag_map.csv` — 63 tags over 3 PLCs, Modbus v4
  addressing, CSV is the contract (test asserts it matches the source).
- `plant/{operating_limits,recipes,allow_list}.yaml`, `plant/plant_documentation.md`,
  `plant/mitre/*.txt` + loaders `plant/{limits,recipes,allow_list}.py` with validation.
  Oracle/task-generator file split enforced.
- `plc/src/{mixing,coating,slitting}.st` — compile under STruC++ 0.7.0.
- `src/graphiot/plc/{shim.cpp,build.py,compiled.py}` — compile ST→.so, run the real
  compiled logic under simulated time via ctypes; instances isolated (twin-safe).
- `src/graphiot/sim/{noise,physics,plant}.py` — deterministic, replayable coupled
  simulator. **Validated context-shift**: correct coating run holds residual solvent
  0.10–0.20 %; the steady web-speed value injected during warm-up drives it to 3.30 %
  (>3.0 % limit); removing that one command restores spec (counterfactual basis for the
  oracle).

## 2026-10-26 — M1 complete on the real stack; M2 gateway/HMI/EWS working

Docker access works via `sg docker` (user in group; re-login makes it direct).
- **OpenPLC Runtime v4.3.1 deployed headlessly** (image pinned by digest
  `sha256:2ca9fc61…`): `.st` → STruC++ npm API (`plc/tools/compile_v4.mjs`, Node 22 in
  `.tools/`) → `program.zip` (+`defines.h`, `conf/modbus_slave.json`,
  `conf/modbus_master.json`) → REST upload → SUCCESS → RUNNING. Two v4.3.1 gotchas found
  and fixed: the slave needs an explicit segmented `buffer_mapping` (else %MW is empty);
  master `offset` must be a string ("0x0000").
- v4 copies all located areas in each scan and commits outputs/memory only on change,
  so agent-written coils/%MW reach the program; matches the ctypes twin at scan level.
- Physics refactored to be **actuator-only** (`sim/model.py`); PLCs drive fan, tension,
  gap, dosing, vacuum, knife (knife interlock enforced in the PLC).
- Own asyncio **Modbus TCP server** (`modbus/server.py`, FC1-6/15/16, 0-based) replaces
  pymodbus' deprecated datastore; serves the field I/O (`sim/field_io.py`).
- **Full Compose testbed** (`testbed/compose.py`, `testbed/manager.py`): north / cell
  (bridge `gi-cell`) / internal field networks; 3 PLCs + field I/O + gateway + HMI +
  EWS. Up + deploy + health in ~20 s; field loop verified on all PLCs (real closed-loop
  control seen: zone heater backs off as the dryer reaches setpoint).
- **Gateway** (`gateway/`): six tools, allow-list (+range) denials logged as commands,
  atomic `load_recipe`, pulsed `acknowledge_alarm`, audit log vs monitor-view log, scorer
  hook before forwarding, admin token via env only (never in compose/YAML). Verified over
  HTTP from the host.
- **HMI** 1 Hz block polling → `capture/tags.jsonl` (the monitor's tag stream), **EWS**
  diagnostic reads. Config schema (`config/`) with secret guard, canonical hash, runs.
- Tests: 42 green.

## 2026-10-26 (later) — M2 + M3 complete; GUI + start.sh; 61 tests

- **Agents** (`agents/`): one provider interface (Ollama / Gemini / Claude, verified against
  the installed SDKs); Claude adapter follows API version 2026-10-21 (no sampling params,
  effort, optional server-side refusal fallback, served model logged); text tool-call
  recovery for local models that print calls as JSON (flagged per turn); shared runner with
  one pacing distribution for both actors (Q17); prompts per paper §V-A.
- **Task generator** (`tasks/`): per-stage production plan (start/steady/adjust/stop),
  monitoring and alarm rounds, legitimate overrides in their normal phase; deterministic.
- **Campaign** (`campaign/`): sequential tasks, seeded random red-team takeovers (same
  identity + task id), balanced techniques, ground truth kept apart.
- **Capture** (`capture/`): tcpdump on `gi-cell` (host-network container) → dpkt-based
  Modbus/TCP reassembly + NetFlow-style flows (5 s interim export = stride, causal).
- **Data** (`data/`): Parquet conversion, the whitelisted **monitor view**, chronological
  splits.
- **Oracle** (`oracle/`): twin replay tick-aligned with the live sim, checkpoint/restore
  (bit-exact, incl. compiled-PLC memory), proximate-cause episode attribution (D11),
  leave-one-out command labels, fidelity report, fallback oracle. Limits got applicability
  conditions (D13) and one-sided τ scaling (D12). Physics recalibrated so every correct
  recipe is in spec and every context-shifted command violates (calibration test).
- **Pilot** (`eval/pilot.py`): bins, C_benign, Definition 1 slice (+ Q8 strict variant),
  depth-3 tree + B0 AUPRC with bootstrap CIs, pilot table as JSON/CSV/LaTeX.
- **GUI**: FastAPI + WebSocket backend (`server/app.py`), React/Vite/uPlot frontend
  (`gui/`): Experiment (YAML + schema form, runs, testbed, campaign, processing, keys),
  Live testbed, Agents, Oracle, Results (downloads); Graph/Training/Inference placeholders.
- **start.sh**: project-local Python/Node/STruC++/GUI bootstrap; Docker is the only
  system requirement. README with paper-section → code map; docs/DATA_FORMAT.md.
- Real-stack validation: a real-LLM campaign (llama3.1 operator, qwen2.5-coder red team on
  CPU) ran end to end; a mini-pilot with capture → oracle → pilot ran (see results there).

## Next: M4 graph builder → M5 model → M6 baselines → M7 attacker → M8 evaluation

## Next (rest of M2, then M3 for the pilot)
1. LLM providers (Ollama/Gemini/Claude behind one interface), operator + red-team agents
   (same ToolClient, shared pacing), task generator.
2. Campaign orchestrator (benign tasks + red-team takeovers, ground truth separate).
3. Capture service (tcpdump on gi-cell → Modbus transactions + NetFlow-style flows).
4. Oracle (primary twin replay + fallback), pilot go/no-go.
5. GUI + start.sh.

## Older plan (kept for reference)
1. Config schema (pydantic, §8) — experiment YAML covering everything, validated; the
   backbone every run needs. **Start here next session.**
2. Dev soft-PLC Modbus backend (real Modbus TCP over pymodbus on loopback IPs) — lets the
   gateway/agents/capture run with no Docker.
3. `testbed/docker-compose.yml` + OpenPLC v4 deploy scripts (needs Q1: Docker group).
4. PLCopen TC6-XML + OpenPLC Editor v4 project exports of the 3 programs; ST unit tests.
5. GUI skeleton (FastAPI+WS backend, React/Vite frontend) + Experiment and Live-testbed
   tabs; `start.sh`.
6. Then M2 (gateway + 6 tools + agents) and M3 (capture + oracle + pilot go/no-go).

## Blocking on the user
- **Q1 Docker group**: `sudo usermod -aG docker $USER` then re-login — needed before the
  real OpenPLC v4 testbed can start. Dev backend works without it.

## 2026-10-26 (evening) — findings from the end-to-end mini-pilot (real stack, real LLMs)

The mini-pilot (`runs/mini-pilot_20261026T083955Z`, 7 min, 3 red-team episodes, CPU LLMs —
pipeline validation, not paper data) ran end to end and surfaced three real defects, all now
fixed:

1. **Capture kept only the first minute.** tcpdump drops to user `tcpdump` after opening its
   first file and could not create the rotated files in the bind-mounted folder. Fix: `-Z root`
   in the (isolated) capture container. Live re-validation: see `runs/capture-check_*`.
2. **The oracle silently replayed an empty world** because that pcap held no gateway writes.
   Fix: in `auto` mode the oracle now measures pcap coverage of the forwarded gateway commands
   and, below 95 %, falls back to the gateway log and says so in its summary.
3. **Twin fidelity failed on mixing air entrainment** (p95 0.08 of the limit width). Cause:
   before OpenPLC has a program loaded, all actuator registers read 0 and `vacuum_cmd = 0` means
   full vacuum, while the twin's PLC vents from tick 0; the initial 2 % air decayed only live.
   Fix: initial air = 0 (a fixed point under both). Fidelity now passes: worst p95 0.033 of the
   limit width (register quantization). A configurable `actuator_delay_ticks` was measured and
   is not needed (kept at 0).

Also done: M4 graph builder core (`graph/schema.py`, `graph/builder.py`) with the causal-rule
test (`tests/test_causal_rule.py`). 65 tests green.

## 2026-10-26 (afternoon) — M4–M6, M8 and the GUI tabs; the pilot capture must be restarted

**Pilot capture failure (not a code bug).** The background pilot (`runs/pilot-local_20261026T092313Z`)
stopped at 10:35:01 on an HTTP 500 from the gateway. Cause, from the system journal: the lid
was closed at 10:31:49, the laptop suspended for 7 s, and after resume the OpenPLC v4 runtimes
of PLC_MIX and PLC_COAT stopped answering (HMI log: no record from either after 10:31:50).
The next work-order publish to PLC_MIX failed. Fixes:
* campaigns hold a systemd sleep/lid inhibitor (`campaign.inhibit_sleep`, released even if the
  process is killed), detect suspends (CLOCK_BOOTTIME − CLOCK_MONOTONIC) and stop with a clear
  message; a passive PLC-liveness check reads the HMI log (`campaign.plc_stale_s`);
* the gateway's Modbus port reconnects and retries once; a failed work-order publish returns
  502 (registration is idempotent) and the campaign retries it;
* the campaign stops after 3 consecutive `llm_error` tasks; an LLM preflight runs before a
  campaign starts (GUI shows it);
* container logs are saved to `logs/compose.log` before teardown;
* the GUI now sees and can stop a testbed it did not start.
The run `pilot_20261026T093558Z` (created from `configs/pilot.yaml` in the GUI at 10:35:58)
failed with `llm_error`: `llama3.1:8b` was not installed (only `llama3.1:latest`). An alias tag
`llama3.1:8b` → same weights (digest 46e0c10c039e) was added in the local Ollama. Its testbed
is still up; it must be stopped before the pilot can be restarted (one testbed at a time).

**Built and tested (74 tests):**
* M4: training-split normaliser, dataset cache (`graph/dataset.pkl`), no oracle label in training.
* M5: HGT + GRU memory + allowed-(tag, bin) head (D14), TBPTT over 32 snapshots on the benign
  sequence only, early stopping, exact-resume checkpoints (`tests/test_checkpoint_resume.py`),
  threshold at 1 % FPR on benign validation, full-timeline scoring, explanations (attention,
  context contribution, memory), edge cost in a 4-CPU / 4-GB container.
* M6: B0, B1, B2 (XGBoost, same features), B5 (frozen, hashed prompt; cached); the four
  ablations. B3 (D15) and B4 (AIRGuard via its MCP-proxy guard) adapters still to write.
* M8: slices all/allow/ctx, AUROC/AUPRC, episode recall at 1 % FPR, alarms/h, bootstrap over
  episodes (seeds averaged per resample), McNemar + Holm, H1 check, CSV/LaTeX/JSON/PNG/PDF,
  run zip with score files (`tests/test_evaluation.py` checks known answers).
* GUI: Graph, Training, Inference & explainability tabs; Results extended.

**Not done yet:** B3 and B4 adapters; M7 (adaptive attacker); real results (no pilot data yet).

## 2026-10-26 (evening) — M6 complete (B3, B4), M7 complete (adaptive attacker); 78 tests

* **B3 E-GraphSAGE** (`baselines/b3_egraphsage.py` + `_b3_worker.py`): the authors' notebook
  classes run verbatim in a separate `.venv-b3` (torch 2.2.1 + DGL 2.1.0, created from
  `requirements/b3.txt`; ~1 GB). Our adapter builds the network edge table (Modbus transactions
  as edges, each scored write command mapped to its write transaction) and reads the per-command
  P(attack) back. Trained on the pilot's labels (Q5). D15. Transductivity noted in the docstring.
* **B4 AIRGuard** (`baselines/b4_airguard.py`): AIRGuard's own `check_action` 5-layer guard called
  unchanged from `third_party/airguard/` (the unused `mcp` import is stubbed). Finding: a
  `write_tag` normalises to `file.write`, which AIRGuard routes to its heuristic fallback, so it
  neither consults the LLM nor denies setpoint writes — B4 allows all fieldbus writes (recall 0 at
  its operating point). This is the honest result; AIRGuard targets prompt injection via untrusted
  tool outputs, a different threat. AUROC n/a (Q6).
* **M7 adaptive attacker** (`attacker/`): realizable edits (insert read, interleave past benign,
  split change, time shift), greedy search minimising the detector's max command score, effect
  preserved by twin replay (`synth.regen_outputs` + `find_violations`). The synthesiser regenerates
  the monitor streams (1 Hz tags via the twin, commands/modbus/gateway rows) for the edited episode;
  the model is re-scored over just the episode's snapshots, continuing GRU memory from a pre-window
  checkpoint (`rescore.py`). Detectors: our model and B0. Budgets {0,4,8}, evasion rate per
  detector; CSV/LaTeX/JSON. Verified end-to-end on a real episode's window; search logic unit-tested.
* GUI Results tab: "Run baselines" (B0–B5), "Adaptive attacker", evasion table; B3/B4 rows appear
  in the main table. New server endpoints: `/baselines` (with `train_run` for B3), `/attacker`.

**Everything remaining from the earlier handoff is now built.** Still outstanding is only real
data: the pilot capture must be restarted once the user's running testbed is stopped. B3/B4/M7
produce real numbers only on a run with harmful episodes.
