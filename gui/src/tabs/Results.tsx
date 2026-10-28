// Results: pilot go/no-go, baselines and evaluation (paper §V-F): main recall table with
// bootstrap CIs, ablations, McNemar + Holm, H1, figures, edge cost. Every number comes
// from files of this run. Downloads: CSV / LaTeX / JSON / PNG / PDF / zip.
import { useState } from "react";
import { api, fmt, post, usePoll } from "../api";

const BASELINES = [["b0", "B0 value table"], ["b1", "B1 allow-list"], ["b2", "B2 XGBoost"], ["b5", "B5 LLM judge (slow: one LLM call per command)"]];

function Table({ rows }: { rows: any[] }) {
  if (!rows?.length) return <div className="muted">–</div>;
  const cols = Object.keys(rows[0]);
  return <div className="scroll" style={{ maxHeight: 420 }}><table><thead><tr>{cols.map((c) => <th key={c}>{c}</th>)}</tr></thead>
    <tbody>{rows.map((r, i) => <tr key={i}>{cols.map((c) => <td key={c} className={c === cols[0] ? "" : "mono"}>{fmt(r[c], 4)}</td>)}</tr>)}</tbody></table></div>;
}

export function Results({ runId }: { runId: string }) {
  const [r, setR] = useState<any>(null);
  const [which, setWhich] = useState<string[]>(["b0", "b1", "b2"]);
  const [resamples, setResamples] = useState("");
  const [msg, setMsg] = useState("");
  usePoll(() => { if (runId) api(`/api/runs/${runId}/results`).then(setR); }, 8000, [runId]);
  const dl = (path: string) => `/api/runs/${runId}/file?path=${encodeURIComponent(path)}`;
  const act = async (path: string, body: any) => { try { await post(path, body); setMsg("started — progress in the Experiment tab → Jobs"); } catch (e: any) { setMsg(e.message); } };
  if (!r) return <div className="placeholder">Loading…</div>;
  const ev = r.evaluation, pilot = r.pilot;
  const figs = (r.files || []).filter((f: string) => f.startsWith("results/figures/") && f.endsWith(".png"));
  return (<div style={{ display: "grid", gap: 14 }}>
    <div className="card"><h3>Run the baselines and the evaluation</h3>
      <div className="row">{BASELINES.map(([k, l]) => <label key={k} className="row"><input type="checkbox" checked={which.includes(k)}
        onChange={(e) => setWhich(e.target.checked ? [...which, k] : which.filter((x) => x !== k))} />{l}</label>)}
        <button onClick={() => act(`/api/runs/${runId}/baselines`, { which })} disabled={!which.length}>Run baselines</button></div>
      <div className="row" style={{ marginTop: 8 }}><span className="muted">bootstrap resamples</span>
        <input value={resamples} onChange={(e) => setResamples(e.target.value)} placeholder="config (2000)" className="mono" style={{ width: 120 }} />
        <button className="primary" onClick={() => act(`/api/runs/${runId}/evaluate`, { resamples: resamples ? +resamples : null })}>Evaluate (metrics, CIs, McNemar, tables, figures)</button>
        <button onClick={() => act(`/api/runs/${runId}/attacker`, { detectors: ["full", "b0"] })}>Adaptive attacker (evasion, B ∈ {"{0,4,8}"})</button></div>
      <div className="muted" style={{ marginTop: 6 }}>B3 (E-GraphSAGE, DGL) and B4 (AIRGuard) run from their public code (tick them above; B3 trains on this run's labels unless a pilot run is set). The model is trained in the Training tab.</div>
      {msg && <div className="muted" style={{ marginTop: 6 }}>{msg}</div>}
    </div>
    {r.attacker && <div className="card"><h3>Adaptive, effect-preserving attacker (paper §IV-C)</h3>
      <div className="muted" style={{ marginBottom: 6 }}>Share of harmful episodes whose maximum command score falls below the detector's 1 % FPR threshold, after a greedy budget of B realizable edits (effect re-checked by the twin oracle).</div>
      <Table rows={r.tables.attacker_table} />
      <div className="row" style={{ marginTop: 8 }}><a href={dl("results/attacker_table.csv")}><button>CSV</button></a>
        <a href={dl("results/attacker_table.tex")}><button>LaTeX</button></a><a href={dl("results/attacker.json")}><button>JSON</button></a></div></div>}
    {ev && <>
      <div className="card"><div className="row"><h3 style={{ margin: 0 }}>H1 — context-shifted slice</h3>
        {ev.h1 ? <span className={`badge ${ev.h1.supported ? "ok" : "bad"}`} style={{ fontSize: 13 }}>{ev.h1.supported ? "supported" : "not supported"}</span> : <span className="badge">needs GRAPHIOT + baselines</span>}</div>
        <div className="muted" style={{ marginTop: 6 }}>Recall at 1 % FPR of GRAPHIOT exceeds each of B0, B2, B3, B4, B5 by ≥ 10 points with Holm-corrected McNemar p &lt; 0.05.
          {ev.h1?.missing_baselines?.length ? ` Missing: ${ev.h1.missing_baselines.join(", ")}.` : ""}</div>
        <div className="row muted" style={{ marginTop: 6 }}>test commands {ev.info.n_test_commands} · harmful {ev.info.n_harmful} · context-shifted {ev.info.n_context_shifted} · benign test {fmt(ev.info.benign_test_hours, 2)} h · {ev.bootstrap_resamples} bootstrap resamples over episodes</div></div>
      <div className="card"><h3>Main table (all · allow · ctx slices; 95 % CI)</h3><Table rows={r.tables.main_table} />
        <div className="row" style={{ marginTop: 8 }}><a href={dl("results/main_table.csv")}><button>CSV</button></a><a href={dl("results/main_table.tex")}><button>LaTeX</button></a></div></div>
      <div style={{ display: "grid", gap: 14 }}>
        <div className="card"><h3>Ablations</h3><Table rows={r.tables.ablation_table} />
          <div className="row" style={{ marginTop: 8 }}><a href={dl("results/ablation_table.csv")}><button>CSV</button></a><a href={dl("results/ablation_table.tex")}><button>LaTeX</button></a></div></div>
        <div className="card"><h3>McNemar (episode decisions, majority vote over seeds) + Holm</h3><Table rows={r.tables.mcnemar} />
          <div className="row" style={{ marginTop: 8 }}><a href={dl("results/mcnemar.csv")}><button>CSV</button></a><a href={dl("results/mcnemar.tex")}><button>LaTeX</button></a></div></div>
      </div>
      {figs.length > 0 && <div className="card"><h3>Figures</h3>
        {figs.map((f: string) => <div key={f} style={{ marginBottom: 10 }}><img src={dl(f)} style={{ width: "100%", maxWidth: 1100, background: "white", borderRadius: 4 }} />
          <div className="row"><a href={dl(f)}><button>PNG</button></a><a href={dl(f.replace(".png", ".pdf"))}><button>PDF</button></a></div></div>)}</div>}
    </>}
    {r.edge_cost && <div className="card"><h3>Edge cost</h3><div className="mono">median {fmt(r.edge_cost.latency_ms_median, 1)} ms · p99 {fmt(r.edge_cost.latency_ms_p99, 1)} ms ·
      peak {fmt(r.edge_cost.peak_rss_mb, 0)} MB · model {fmt(r.edge_cost.model_kb, 0)} kB — {r.edge_cost.environment}</div></div>}
    {pilot && <div className="card">
      <div className="row"><h3 style={{ margin: 0 }}>Pilot go / no-go (paper §V-C)</h3>
        <span className={`badge ${pilot.table.decision === "GO" ? "ok" : "bad"}`} style={{ fontSize: 14 }}>{pilot.table.decision}</span></div>
      <div className="grid3" style={{ marginTop: 10 }}>
        <div><div className="muted">context-shifted commands (≥ 50 needed)</div><div className="kpi">{pilot.test.n_context_shifted}</div>
          <div className="muted">{pilot.test.n_context_shifted_strict} within the benign value range (Q8)</div></div>
        <div><div className="muted">depth-3 tree AUPRC (&lt; 0.9 needed)</div><div className="kpi">{pilot.test.tree_auprc == null ? "–" : fmt(pilot.test.tree_auprc, 3)}</div>
          <div className="muted">{pilot.test.tree_auprc_ci95 ? `95 % CI ${fmt(pilot.test.tree_auprc_ci95[0], 3)}–${fmt(pilot.test.tree_auprc_ci95[1], 3)}` : pilot.test.note}</div></div>
        <div><div className="muted">B0 phase-value table AUPRC</div><div className="kpi">{pilot.test.b0_auprc == null ? "–" : fmt(pilot.test.b0_auprc, 3)}</div>
          <div className="muted">test prevalence {fmt(pilot.test.prevalence_test, 3)} · {pilot.test.pos_test}/{pilot.test.n_test} positives</div></div>
      </div>
      <table style={{ marginTop: 10 }}><tbody>
        {Object.entries(pilot.table).filter(([k]) => k !== "per_technique").map(([k, v]: any) => <tr key={k}><td>{k.replaceAll("_", " ")}</td><td className="mono">{fmt(v, 3)}</td></tr>)}
        {Object.entries(pilot.table.per_technique || {}).map(([t, d]: any) => <tr key={t}><td>technique {t} — episodes / harmful / context-shifted</td><td className="mono">{d.episodes} / {d.harmful} / {d.context_shifted}</td></tr>)}
      </tbody></table>
      <div className="row" style={{ marginTop: 10 }}>
        <a href={dl("results/pilot_table.csv")}><button>CSV</button></a><a href={dl("results/pilot_table.tex")}><button>LaTeX</button></a>
        <a href={dl("results/pilot.json")}><button>JSON</button></a><a href={dl("oracle/command_labels.parquet")}><button>Command labels (CSV)</button></a></div></div>}
    <div className="card"><div className="row"><h3 style={{ margin: 0 }}>Downloads</h3><span className="spacer" />
      <a href={`/api/runs/${runId}/bundle.zip`}><button className="primary">Run bundle (zip: config, oracle, scores, results)</button></a></div>
      <div className="row" style={{ marginTop: 8, flexWrap: "wrap" }}>{(r.files || []).filter((f: string) => !f.endsWith(".png")).map((f: string) =>
        <a key={f} href={dl(f)} className="mono" style={{ fontSize: 11 }}>{f.replace("results/", "")}</a>)}</div></div>
  </div>);
}
