// Inference & explainability: every command's score, threshold and alarm for any detector;
// for our model, why a command scored what it did (prediction, context contribution,
// HGT attention per layer/head/relation, GRU memory of the context nodes); edge cost.
import { useEffect, useMemo, useState } from "react";
import { api, fmt, fmtTime, post, usePoll } from "../api";
import { Chart } from "../components/Chart";
import { CyGraph, Legend } from "../components/CyGraph";

const MEM_COLORS = ["#3987e5", "#d95926", "#199e70", "#c98500"];

export function Inference({ runId }: { runId: string }) {
  const [dets, setDets] = useState<string[]>([]);
  const [det, setDet] = useState("");
  const [data, setData] = useState<any>(null);
  const [filter, setFilter] = useState("test+redteam");
  const [onlyAlarms, setOnlyAlarms] = useState(false);
  const [cmd, setCmd] = useState<number | null>(() => { const c = new URLSearchParams(location.search).get("cmd"); return c === null ? null : +c; });
  const [ex, setEx] = useState<any>(null);
  const [exErr, setExErr] = useState("");
  const [layer, setLayer] = useState(1);
  const [res, setRes] = useState<any>(null);
  useEffect(() => { setDets([]); setDet(""); setData(null);
    if (runId) api(`/api/runs/${runId}/detectors`).then((d) => { const n = d.detectors.filter((x: string) => !x.endsWith("_memory"));
      setDets(n); setDet(n.find((x: string) => x.startsWith("full_seed")) || n[0] || ""); }); }, [runId]);
  useEffect(() => { try { const u = new URL(location.href); if (cmd === null) u.searchParams.delete("cmd"); else u.searchParams.set("cmd", String(cmd));
    history.replaceState(null, "", u); } catch { /* ignore */ } }, [cmd]);
  useEffect(() => { if (runId && det) api(`/api/runs/${runId}/scores/${det}`).then(setData).catch(() => setData(null)); }, [runId, det]);
  usePoll(() => { if (runId) api(`/api/runs/${runId}/results`).then(setRes); }, 10000, [runId]);
  const seed = det.startsWith("full_seed") ? det.split("_seed")[1] : undefined;
  useEffect(() => {
    setEx(null); setExErr("");
    if (cmd === null || !runId) return;
    api(`/api/runs/${runId}/explain/${cmd}${seed ? `?seed=${seed}` : ""}`).then(setEx).catch((e) => setExErr(e.message));
  }, [cmd, runId, seed]);
  const rows = useMemo(() => (data?.commands || []).filter((c: any) =>
    (filter === "all" || filter.split("+").includes(c.split)) && (!onlyAlarms || c.alarm)), [data, filter, onlyAlarms]);
  const graph = useMemo(() => {
    if (!ex) return null;
    const nodes = Object.entries(ex.nodes || {}).flatMap(([t, ks]: any) => ks.map((k: string) => ({ id: k, type: t, label: k.split(":").slice(1).join(":") })));
    const edges = (ex.attention || []).filter((a: any) => a.layer === layer).map((a: any, i: number) =>
      ({ id: `e${i}`, source: a.src, target: a.dst, relation: a.relation.split("__")[1], weight: a.alpha }));
    return { nodes, edges };
  }, [ex, layer]);
  const mem = useMemo(() => {
    if (!ex?.memory_trajectory?.length) return null;
    const tr = ex.memory_trajectory;
    return { ts: tr.map((r: any) => r.T), series: ex.watch.map((w: string, i: number) => ({ label: w, color: MEM_COLORS[i], values: tr.map((r: any) => r[w]?.norm ?? null) })) };
  }, [ex]);
  const ec = res?.edge_cost;
  return (<div style={{ display: "grid", gap: 14 }}>
    <div className="card"><div className="row">
      <h3 style={{ margin: 0 }}>Scores</h3>
      <select value={det} onChange={(e) => { setDet(e.target.value); setCmd(null); }}>{dets.map((d) => <option key={d}>{d}</option>)}</select>
      <select value={filter} onChange={(e) => setFilter(e.target.value)}>
        {["test+redteam", "redteam", "test", "val", "train", "all"].map((f) => <option key={f}>{f}</option>)}</select>
      <label className="row"><input type="checkbox" checked={onlyAlarms} onChange={(e) => setOnlyAlarms(e.target.checked)} /> alarms only</label>
      <span className="spacer" /><span className="mono">threshold {fmt(data?.threshold, 3)}</span>
      <span className="muted">{rows.length} commands · {rows.filter((c: any) => c.alarm).length} alarms</span></div>
      {!dets.length ? <div className="muted" style={{ marginTop: 8 }}>No scores yet: train the model (Training tab) or run the baselines (Results tab).</div> :
      <div className="scroll" style={{ maxHeight: 300, marginTop: 8 }}><table>
        <thead><tr><th>time</th><th>split</th><th>task</th><th>type</th><th>tool</th><th>tag</th><th>value</th><th>phase</th><th>gateway</th><th>score</th><th>alarm</th><th>oracle</th></tr></thead><tbody>
          {rows.map((c: any) => <tr key={c.command_idx} onClick={() => setCmd(c.command_idx)} style={{ cursor: "pointer", background: c.command_idx === cmd ? "var(--panel2)" : undefined }}>
            <td className="mono">{fmtTime(c.ts)}</td><td>{c.split}</td><td className="mono">{c.task_id}</td><td>{c.task_type}</td><td>{c.tool}</td>
            <td className="mono">{c.tag}</td><td className="mono">{c.value}</td><td>{c.phase}</td><td>{c.decision}</td>
            <td className="mono">{c.score === null ? "∞" : fmt(c.score, 3)}</td><td>{c.alarm ? <span className="badge bad">alarm</span> : ""}</td>
            <td>{c.harmful === undefined || c.harmful === null ? "–" : c.harmful ? <span className="badge bad">{c.context_shifted ? "harmful · ctx" : "harmful"}</span> : <span className="badge ok">benign</span>}</td></tr>)}
        </tbody></table></div>}
    </div>
    {cmd !== null && <div className="card">
      <h3>Why this score — command {cmd}{seed ? ` (GRAPHIOT seed ${seed})` : ""}</h3>
      {exErr && <div className="err">{exErr}</div>}
      {!ex && !exErr && <div className="muted">Replaying the memory up to the command's snapshot…</div>}
      {ex && <div style={{ display: "grid", gap: 12 }}>
        <div className="grid3">
          <div><div className="muted">surprise −log p(tag, bin)</div><div className="kpi">{ex.score === null ? "∞" : fmt(ex.score, 3)}</div>
            <div className="muted">{ex.in_allowed_set ? `p = ${ex.observed.p.toExponential(2)} for ${ex.observed.tag}, bin ${ex.observed.bin}` : "tag outside the task's allowed set (p = 0)"}</div></div>
          <div><div className="muted">snapshot used (causal rule)</div><div className="kpi">k = {ex.snapshot.k}</div>
            <div className="muted">T = {fmtTime(ex.snapshot.T)} ≤ t = {fmtTime(ex.command.ts)}</div></div>
          <div><div className="muted">context contribution (Δ surprise when an input is removed)</div>
            {Object.entries(ex.context_contribution).map(([k, v]: any) => <div key={k} className="row" style={{ gap: 6 }}>
              <span style={{ width: 44 }} className="muted">{k}</span>
              <svg width="160" height="10"><rect x={v === null || v >= 0 ? 80 : 80 + Math.max(-80, v * 8)} width={v === null ? 0 : Math.min(80, Math.abs(v) * 8)} height="10" rx="2" fill={v !== null && v >= 0 ? "#d95926" : "#3987e5"} /><line x1="80" x2="80" y1="0" y2="10" stroke="#8b97a8" /></svg>
              <span className="mono">{v === null ? "not used" : (v >= 0 ? "+" : "") + fmt(v, 2)}</span></div>)}</div>
        </div>
        <div className="grid2">
          <div><div className="row" style={{ marginBottom: 6 }}><b>Attention, layer</b>
            {[0, 1].map((l) => <button key={l} className={l === layer ? "primary" : ""} onClick={() => setLayer(l)}>{l + 1}</button>)}
            <span className="muted">edge width = mean attention over heads · white ring = the head's context nodes</span></div>
            {graph && <CyGraph nodes={graph.nodes} edges={graph.edges} showStatic focus={ex.watch} height={420} />}<div style={{ marginTop: 6 }}><Legend /></div></div>
          <div style={{ display: "grid", gap: 10, alignContent: "start" }}>
            <div><b>Most likely (tag, bin) for this task and context</b><table><tbody>
              {ex.top.map((t: any, i: number) => <tr key={i}><td className="mono">{t.tag}</td><td>bin {t.bin}</td><td className="mono">{fmt(t.p, 4)}</td></tr>)}</tbody></table></div>
            <div><b>Attention into the context nodes</b><div className="scroll" style={{ maxHeight: 220 }}><table>
              <thead><tr><th>layer</th><th>relation</th><th>from → to</th><th>α (mean)</th><th>α per head</th></tr></thead><tbody>
              {[...ex.attention_into_context].sort((a: any, b: any) => b.alpha - a.alpha).slice(0, 40).map((a: any, i: number) =>
                <tr key={i}><td>{a.layer + 1}</td><td className="mono">{a.relation.split("__")[1]}</td><td className="mono">{a.src} → {a.dst}</td>
                  <td className="mono">{fmt(a.alpha, 3)}</td><td className="mono">{a.alpha_heads.map((h: number) => h.toFixed(2)).join(" ")}</td></tr>)}</tbody></table></div></div>
          </div>
        </div>
        {mem && <div><b>GRU memory of the context nodes (L2 norm, last {ex.memory_trajectory.length} snapshots)</b>
          <Chart ts={mem.ts} series={mem.series} height={180} /></div>}
      </div>}
    </div>}
    <div className="card"><div className="row"><h3 style={{ margin: 0 }}>Edge cost (paper §IV-B)</h3><span className="spacer" />
      <button onClick={() => post(`/api/runs/${runId}/edge_cost`)}>Measure (gateway limits from the config)</button></div>
      {ec ? <table style={{ marginTop: 8 }}><tbody>
        <tr><td>environment</td><td className="mono">{ec.environment}</td></tr>
        <tr><td>latency per command, median / p99 (snapshot update + scoring)</td><td className="mono">{fmt(ec.latency_ms_median, 1)} ms / {fmt(ec.latency_ms_p99, 1)} ms</td></tr>
        <tr><td>scoring only, median / p99</td><td className="mono">{fmt(ec.score_ms_median, 2)} ms / {fmt(ec.score_ms_p99, 2)} ms</td></tr>
        <tr><td>snapshot update, median / p99</td><td className="mono">{fmt(ec.update_ms_median, 1)} ms / {fmt(ec.update_ms_p99, 1)} ms</td></tr>
        <tr><td>peak resident memory</td><td className="mono">{fmt(ec.peak_rss_mb, 0)} MB</td></tr>
        <tr><td>model size · parameters</td><td className="mono">{fmt(ec.model_kb, 0)} kB · {ec.params?.toLocaleString()}</td></tr>
        <tr><td>commands · snapshots measured</td><td className="mono">{ec.n_commands} · {ec.n_snapshots}</td></tr></tbody></table>
        : <div className="muted" style={{ marginTop: 8 }}>Not measured yet for this run.</div>}
    </div>
  </div>);
}
