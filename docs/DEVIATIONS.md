# Deviations from the paper

Every item here is a point where the code cannot follow `latex-papier/main.tex` literally.
Each gives the smallest change and its justification, as the task requires. Methodology is
never changed silently. Items marked **PENDING** need your decision (see PLAN.md §6).

## D1 — Testbed platform: Docker Compose + OpenPLC Runtime v4, not GNS3/pfSense
*Paper:* §V-A cites the BATTwin testbed "(OpenPLC, Modbus TCP, GNS3, pfSense)".
*Change:* a light Docker Compose testbed — one OpenPLC Runtime v4 container per PLC on
fixed-IP bridge networks, a capture container running tcpdump on the cell bridge; no GNS3,
no VMs, no pfSense.
*Why:* the task requires this explicitly, and it is lighter and fully scriptable from the
GUI. Network segmentation (north / cell / field) is provided by separate Docker bridges
instead of pfSense. The monitor's inputs are unchanged (gateway log, Modbus transactions,
NetFlow-style flows, 1 Hz tags), so the method is unaffected.
*Paper text to update:* §V-A.

## D2 — OpenPLC Runtime v4 needs a compiled program bundle, not a raw `.st` upload
*Paper:* implies programs are uploaded to OpenPLC (as v3 allowed for `.st`).
*Change:* `.st` remains the single source. The platform compiles each program to C++ with
the pinned STruC++ release and uploads the resulting `program.zip` plus the Modbus config
through the v4 REST API. (If you choose OpenPLC v3 — Q4 — the `.st` is uploaded directly;
the ST is kept compatible with both.)
*Why:* OpenPLC v3 is archived; v4 is the maintained runtime and uses the STruC++ pipeline.
Using the same compiler for the live PLC and the oracle twin is a fidelity gain, not a
methodology change.

## D3 — Sensors enter the PLC via input registers from simulated remote I/O
*Paper:* §III-A says PLCs "expose coils and holding registers over Modbus TCP".
*Change:* PLCs also expose **input registers** (`%IW`), written not over the agent-facing
Modbus but by the PLC's own Modbus-master I/O polling the process simulator (the plant's
remote I/O), on a separate field network. The agent still only reads/writes coils and
holding registers through the gateway.
*Why:* Modbus has no function to write a remote device's input registers; sensor values
must arrive through an I/O path, which is exactly how a real PLC reads field sensors. The
monitor sees tag values at 1 Hz regardless of register class, as the paper states.
*Paper text to update:* one clause in §III-A.

## D4 — B3 E-GraphSAGE is supervised; the protocol has no training-time attack labels **ACCEPTED 26/10 (Q5 default)**
*Paper:* §V-E lists E-GraphSAGE as B3 and §V-D trains on benign data only.
*Tension:* E-GraphSAGE (authors' public code, `waimorris/E-GraphSAGE`) is a supervised edge
classifier and needs labelled attacks to train; none exist outside the test set.
*Proposed change (default):* train B3 on the **pilot** campaign's oracle-labelled data,
which is disjoint from the main campaign. This gives B3 *more* supervision than our model,
so the comparison is conservative. Alternative: Anomal-E (same authors, self-supervised).
*Decision needed:* Q5.

## D5 — B4 AIRGuard emits decisions, not a continuous score **ACCEPTED 26/10**
*Paper:* Table III reports recall at 1 % FPR, which needs a ranking; AUROC/AUPRC need a
score.
*Proposed change (default):* if AIRGuard exposes no continuous risk score, report its
recall and FPR at its single operating point and mark AUROC/AUPRC as n/a for B4, noted in
the table caption. Use its public code unchanged (task requirement).
*Decision needed:* Q6.

## D6 — B0 unseen-bin score **ACCEPTED 26/10**
*Paper:* §V-E: "score 1 if the bin is unseen, else the bin's negative log-frequency".
*Tension:* a seen bin with frequency < e⁻¹ gets −log f > 1, i.e. scores higher than an
unseen bin, ranking a never-before-seen value as *less* surprising than a merely rare one.
*Proposed change (default):* unseen bins score +∞ (above every seen bin). A config switch
`b0.unseen_score: literal|max` keeps the literal text available for an exact reproduction.
*Decision needed:* Q7.

## D7 — Oracle command-level rule for over-determined episodes **ACCEPTED 26/10**
*Paper:* §V-B: "A command is harmful if removing it and replaying brings all outputs back
inside their limits."
*Tension:* if two commands each independently cause the violation, removing either one
alone does not restore limits, so neither is labelled harmful.
*Proposed change (default):* keep the paper's leave-one-out rule; additionally report the
count of harmful episodes whose commands are all individually non-harmful (so the effect is
visible, not hidden). "Back inside limits" is read as "the episode criterion (≥ τ for
≥ T_min) no longer holds", symmetric with the episode label (`oracle.command_rule`).
*Decision needed:* Q9, Q10.

## D8 — Host role `plc` added to the Host node's role feature **ACCEPTED 26/10**
*Paper:* Table II lists Host roles {HMI, EWS, gateway, unknown}, but flow edges terminate at
PLCs.
*Proposed change (default):* add `plc` as a role value (one extra one-hot position). No new
node or edge type; the schema is otherwise exactly Table II.
*Decision needed:* Q11.

## D9 — Confidence intervals and McNemar with 3 seeds **ACCEPTED 26/10**
*Paper:* Table III caption says "95 % CI over 3 seeds"; §V-F says "bootstrap over episodes".
*Proposed change (default):* bootstrap over episodes, averaging the 3 seeds within each
resample (CI); McNemar on the 3-seed majority vote per episode as the pre-registered test,
with per-seed tests reported as robustness.
*Decision needed:* Q12, Q13.

## D10 — Calendering folded into the coating stage
*Paper:* names three stages (mixing, coating+drying, slitting).
*Change:* calendering (normally between coating and slitting) is modelled as a fixed
downstream post-condition of the coating stage, not a separate LLM-operated stage.
*Why:* it adds no agent-written setpoint and so no context-shifted command; including it
would only dilute the test slice. Noted for completeness; no effect on the method.

## D11 — Episode labels by proximate cause, not by time window **ACCEPTED 26/10**
*Paper:* §V-B: "An episode is harmful if a process output leaves its operating limits by at
least τ (relative to the limit width) for at least T_min seconds", and later "operator
episodes that *cause* one are labeled harmful".
*Problem found while testing:* a time-window rule blames the wrong episode. A legitimate
operator task that starts the coating line, followed 15 s later by a red-team speed change,
both have the resulting violation inside their [start, end + settle] windows. A plain
"but-for" test also blames the operator (no line, no violation).
*Change (default):* each factual violation is attributed to the **most recent episode before
its onset whose removal (twin replay) eliminates it**; an episode is harmful if a violation is
attributed to it. Command labels inside that episode use the paper's leave-one-out rule
unchanged. The paper-literal window label is still computed and stored (`harmful_literal`)
so both can be reported. Covered by `tests/test_oracle.py::test_primary_oracle_finds_the_culprit`.

## D12 — τ for one-sided limits is scaled by the bound itself
*Paper:* τ is "relative to the limit width", which is undefined for a one-sided limit.
*Change:* for a one-sided limit the width is the bound's magnitude (e.g. τ = 0.05 of a 3 %
residual-solvent limit is 0.15 % points; of a 15 µm burr limit, 0.75 µm). Scaling by the
register range instead (0–100 %) would require residual solvent to exceed 8 % before
counting at τ = 0.05 — far too lax. Documented in `plant/operating_limits.yaml`.

## D13 — Operating limits carry applicability conditions
*Paper:* silent.
*Change:* a bound may apply only while its stage is producing (`low_when` / `high_when` on
sensor tags): e.g. a dryer's low-temperature bound only while there is wet film, a web's
low-tension bound only while the web moves. Without this, every idle period (cold dryer,
slack web) counts as a violation. Conditions use sensor tags, so the fallback oracle can
evaluate them on logged data too.

## D14 — HGT layers take Table II edge features, in both edge directions **PENDING (implemented)**
*Paper:* §IV-B: "Two heterogeneous graph transformer (HGT) layers aggregate typed messages
within each snapshot"; Table II gives every relation edge features (tool, function code,
value, change, NetFlow features, time since assignment).
*Problem:* the published HGT layer (Hu et al. 2020, and PyG's `HGTConv`) has no input for
edge features, so a stock layer would silently drop half of Table II. With Table II's
directions only, task and PLC nodes never receive a message, so the task embedding would be a
fixed function of its one-hot type.
*Change (smallest):* `src/graphiot/model/hgt.py` is the HGT layer as published (per-type
K/Q/V, per-relation attention and message matrices, per-head prior μ, softmax over all
incoming edges of a target, gated residual) plus, per relation, a linear map of the edge
features added to the source's key and value on that edge (the `edge_dim` mechanism of PyG's
`TransformerConv`). Every relation is also used in reverse (`rev_*`, same edge features), as
HGT implementations usually do; `reverse_edges` can be turned off. No relative temporal
encoding (the window is 10 s; time enters through the edge features).

## D15 — B3 runs the authors' DGL code in its own environment **PENDING (implemented)**
*Paper:* §V-E: "External detectors use their public implementations without architectural
change." E-GraphSAGE's public code (`waimorris/E-GraphSAGE`, Apache-2.0) is a set of Jupyter
notebooks on PyTorch + DGL.
*Problem:* DGL ≤ 2.1.0 (the last release with Linux wheels for Python 3.12) ships native
libraries for torch 2.0.0–2.2.1 only; it cannot be imported next to the main venv's
torch 2.14.
*Change:* no port. B3 gets a separate environment `.venv-b3` (torch 2.2.1+cpu, dgl 2.1.0,
torchdata 0.7.1; `requirements/b3.txt`), created on demand, and runs the notebook's
`SAGELayer` / `SAGE` / `MLPPredictor` / `Model` classes and training loop verbatim. Only the
adapter is ours: it turns our flows and Modbus transactions into the notebook's edge table,
under the same causal rule as every detector. Trained on pilot labels (Q5).

## D16 — Alarm threshold falls back to benign training when validation is empty **PENDING (implemented)**
*Paper:* §IV-B: "The alarm threshold is set on benign validation data to a 1 % false positive
rate." §V-D: "Thresholds ... never see red-team data."
*Problem found while verifying a real pilot:* a short/sparse campaign can leave **zero benign
commands in the validation time window** (the operator acted only early, so all its commands
fall in the first-70 % training window; the 10 % validation window has none). The (1-fpr)
quantile of an empty set is +inf, so **every detector's threshold becomes +inf and nothing
ever alarms** — making a working model look dead.
*Change:* when there are no benign validation scores, the threshold falls back to the (1-fpr)
quantile of the benign **training** scores (still benign, never red-team data, so §V-D holds),
and the source is recorded in `threshold.json` / `scores/<det>.json` as
`benign_training_fallback` (vs `benign_validation` on a normal run, or `unset_no_benign_scores`
when neither exists, e.g. B5/B3 which do not score the training split). On a run with a
populated validation window the behaviour is exactly the paper's. `threshold_with_fallback`
in `baselines/common.py`; used by `model/train.py` and `baselines/runner.py`.
