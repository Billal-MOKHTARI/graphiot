# GRAPHIOT — experiment platform

Code and GUI for the experiments of *“Detecting Harmful LLM-Agent Commands in Industrial
IoT with Task-Aware Graph Neural Networks”* (IEEE IoT Journal special issue; paper source:
`latex-papier/main.tex`). A light battery-electrode testbed (three real OpenPLC Runtime v4
controllers, real Modbus TCP), an edge gateway with six tools, LLM operator and red-team
agents, a pre-registered-style oracle on process outputs, and — milestone by milestone — the
task-aware GNN, the baselines, the adaptive attacker and the evaluation.

> Status (26/10/2026): M1 testbed, M2 gateway + agents, M3 capture + oracle + pilot test are
> built and tested on the real stack. M4–M8 (graph, model, baselines, attacker, evaluation)
> are next. See `docs/PROGRESS.md`.

## Requirements

**Docker** is the only system requirement (Linux; Docker Engine 25+ with Compose v2). Your
user must be able to run `docker` (`sudo usermod -aG docker $USER`, then log out and in).
Everything else — Python 3.12, Node 22, the STruC++ PLC compiler, every package — is
installed inside this folder (`.tools/`, `.venv/`, `gui/node_modules/`) by `start.sh`.
Free disk: ~5 GB (OpenPLC image ~1 GB, services image ~0.4 GB, venv ~1.5 GB).

## Launch

```bash
./start.sh            # first run sets everything up, then opens http://127.0.0.1:8765
```

Then, from the GUI:

1. **Experiment** — pick or upload a config (`configs/pilot.yaml`), edit it as YAML or in
   the generated form, *Validate*, *Create run*.
2. *Testbed up* — compose up, compile and deploy the three PLC programs, start capture.
3. *Start campaign* — benign operation with random red-team takeovers (watch **Live
   testbed** and **Agents**).
4. *Testbed down*, then *Oracle + pilot test* — Parquet tables, twin-replay oracle, pilot
   go/no-go (**Oracle**, **Results**, downloads as CSV / JSON / LaTeX / zip).

API keys (Gemini / Claude) go in `.env` (see `.env.example`) or the GUI's key field; they
stay in memory and are never written to YAML, logs or exports. Local Ollama needs none; for
the pilot point `llm.*.base_url` at the GPU server.

## How the pieces fit

```
agents (LLM) --HTTP--> gateway --Modbus--> PLC_MIX / PLC_COAT / PLC_SLIT (OpenPLC v4)
   north net             cell net (captured)          | field net (remote I/O)
                                                      +--> process simulator (physics)
oracle twin = same physics + the same ST compiled by STruC++, replayed offline
```

## Paper section → code

| Paper | Code |
|---|---|
| §III-A cell, three stages, OpenPLC, tags | `plc/src/*.st`, `plant/tag_map.csv`, `src/graphiot/plant/tags.py`, `testbed/` |
| §III-A gateway, six tools, allow-list, log | `src/graphiot/gateway/` (`core.py`, `tools_spec.py`), `plant/allow_list.yaml` |
| §III-B defender model (what the monitor sees) | `src/graphiot/data/monitor_view.py`, `tests/test_no_leakage.py` |
| §III-D context-shifted commands (Def. 1), bins | `src/graphiot/eval/bins.py`, `src/graphiot/eval/pilot.py` |
| §IV-A task–plant graph, causal rule | `src/graphiot/graph/builder.py` (snapshots), `graph/schema.py` (Table II), `graph/dataset.py` (normaliser on the training split), `tests/test_causal_rule.py` |
| §IV-B model, training signal, score, threshold | `src/graphiot/model/hgt.py` (HGT layer, D14), `model/net.py` (projection, GRU memory, (tag, bin) head, ablation variants), `model/train.py` (TBPTT, early stopping, exact-resume checkpoints, threshold, scoring), `model/explain.py`, `tests/test_checkpoint_resume.py` |
| §IV-B edge deployment (latency, memory, size) | `src/graphiot/model/edge_cost.py` (capped container) |
| §IV-C adaptive attacker | `src/graphiot/attacker/`: `core.py` (episodes, edits, greedy search, evasion), `synth.py` (regenerate monitor streams via the twin), `rescore.py` (model re-score), `rescore_b0.py` (B0), `tests/test_attacker.py` |
| §V-A capture, operator & red-team agents, tasks | `src/graphiot/campaign/`, `src/graphiot/agents/`, `src/graphiot/tasks/`, `src/graphiot/capture/` |
| §V-B oracle (primary twin replay, fallback) | `src/graphiot/oracle/core.py`, `src/graphiot/sim/`, `plant/operating_limits.yaml` |
| §V-C pilot go/no-go | `src/graphiot/eval/pilot.py` |
| §V-D chronological splits | `src/graphiot/data/splits.py`, `tests/test_splits.py` |
| §V-E baselines | `src/graphiot/baselines/`: B0 `b0.py`, B1 `b1.py` (the gateway's policy), B2 `b2.py` (XGBoost, same features), B3 `b3_egraphsage.py` (authors' DGL code in `.venv-b3`, D15), B4 `b4_airguard.py` (AIRGuard unchanged, `third_party/`), B5 `b5.py` + frozen `b5_prompt.txt` |
| §V-E ablations | `src/graphiot/model/net.py` `VARIANTS` (agent-only, plant-only, network-only, no task node) |
| §V-F metrics, statistics | `src/graphiot/eval/evaluate.py` (slices, recall at 1 % FPR, alarms/h, bootstrap, McNemar + Holm, H1, tables), `eval/metrics.py`, `eval/figures.py`, `tests/test_evaluation.py` |
| process design, literature | `docs/PROCESS_DESIGN.md` |
| deviations from the paper | `docs/DEVIATIONS.md` |

## Baselines B3 and B4 (public code)

B0–B2 and B5 are in `src/graphiot/baselines/` and run in the main environment. B3 and B4
run the authors' public code unchanged:

- **B3 E-GraphSAGE** (`waimorris/E-GraphSAGE`, Apache-2.0) is DGL-based and cannot share the
  main venv (DGL has no build for torch 2.14), so it runs in a separate `./.venv-b3`
  (torch 2.2.1 + DGL 2.1.0). Create it once:
  ```bash
  uv venv --python .tools/python/*/bin/python3.12 .venv-b3
  uv pip install --python .venv-b3/bin/python \
      --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple \
      --index-strategy unsafe-best-match "torch==2.2.1+cpu" -r requirements/b3.txt
  ```
  B3 trains on a labelled run (the pilot, Q5) and scores the main run; see `docs/DEVIATIONS.md` D15.
- **B4 AIRGuard** (`Sophie508/AIRGuard`, MIT) is cloned under `third_party/airguard/`; its 5-layer
  `check_action` guard is called unchanged (the unused `mcp` server import is stubbed). It returns
  decisions, not scores (Q6): recall/FPR at one operating point, AUROC n/a.

## Tests

```bash
.venv/bin/python -m pytest -q
```

Covers the tag map, plant data split (oracle vs task generator), PLC compilation and
twin-instance isolation, simulator determinism and counterfactual replay, the calibration
premise (every correct recipe in spec; every context-shifted command violates), the custom
Modbus server, the v4 bundle, the config schema (secrets refused, families differ), oracle
labels, chronological splits and no leakage into the monitor.

## Reproducibility

Each run folder (`runs/<name>_<utc>/`) stores the exact config, its SHA-256, the git commit
(+ dirty flag and a source-tree hash), seeds, package versions, the rendered
`docker-compose.yml`, the deployed PLC bundles with their MD5, and every log. Data format:
`docs/DATA_FORMAT.md`. Pinned versions: `requirements/*.txt`, `gui/package.json`,
`plc/tools/package.json`, OpenPLC image digest and STruC++ release in the config/`start.sh`.
