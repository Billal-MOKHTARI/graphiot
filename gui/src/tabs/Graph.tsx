// Graph: the task–plant snapshots (paper §IV-A, Table II) — timeline slider over T_k,
// normalised (what the model sees) or raw (engineering units) features, features on click.
import { useEffect, useMemo, useState } from "react";
import { api, fmtTime, post } from "../api";
import { CyGraph, FeatureTable, Legend } from "../components/CyGraph";

const SPLIT_COLOR: Record<string, string> = { train: "#2bb39b", val: "#6ea8fe", test: "#f2a93b", redteam: "#f0544f", excluded: "#4a5566" };

export function Graph({ runId }: { runId: string }) {
  const [meta, setMeta] = useState<any>(null);
  const [k, setK] = useState(0);
  const [snap, setSnap] = useState<any>(null);
  const [raw, setRaw] = useState(false);
  const [showStatic, setShowStatic] = useState(false);
  const [sel, setSel] = useState<any>(null);
  const [msg, setMsg] = useState("");
  const load = () => runId && api(`/api/runs/${runId}/graph/meta`).then((m) => { setMeta(m); if (m.available) setK((x) => Math.min(x, m.n - 1)); });
  useEffect(() => { setMeta(null); setSnap(null); load(); }, [runId]);
  useEffect(() => {
    if (!meta?.available) return;
    const t = setTimeout(() => api(`/api/runs/${runId}/graph/snapshot/${k}?raw=${raw}`).then(setSnap).catch((e) => setMsg(e.message)), 120);
    return () => clearTimeout(t);
  }, [k, raw, meta?.available, runId]);
  const build = async () => { setMsg("building… (Jobs in the Experiment tab)"); await post(`/api/runs/${runId}/graph/build`); setTimeout(load, 4000); };
  const strip = useMemo(() => meta?.available ? meta.splits.map((s: string, i: number) => ({ s, i })) : [], [meta]);
  if (!meta) return <div className="placeholder">Loading…</div>;
  if (!meta.available) return <div className="card placeholder">
    <p>No graph dataset for this run yet{meta.has_capture ? "" : " (run a campaign and “Oracle + pilot test” first, which builds the capture tables)"}.</p>
    <button className="primary" disabled={!meta.has_capture} onClick={build}>Build snapshots</button> <span className="muted">{msg}</span></div>;
  return (<div style={{ display: "grid", gap: 14 }}>
    <div className="card">
      <div className="row"><h3 style={{ margin: 0 }}>Snapshot {k} / {meta.n - 1}</h3>
        {snap && <><span className="mono">T = {fmtTime(snap.T)}</span><span className="badge" style={{ color: SPLIT_COLOR[snap.split] }}>{snap.split}</span>
          <span className="muted">{snap.nodes.length} nodes · {snap.edges.length} edges · window Δ = {meta.window_s} s, stride {meta.stride_s} s</span></>}
        <span className="spacer" />
        <label className="row"><input type="checkbox" checked={raw} onChange={(e) => setRaw(e.target.checked)} /> raw units</label>
        <label className="row"><input type="checkbox" checked={showStatic} onChange={(e) => setShowStatic(e.target.checked)} /> static edges</label>
        <button onClick={build}>Rebuild</button>
      </div>
      <input type="range" min={0} max={meta.n - 1} value={k} onChange={(e) => setK(+e.target.value)} style={{ width: "100%", marginTop: 8 }} />
      <svg width="100%" height="10" viewBox={`0 0 ${meta.n} 1`} preserveAspectRatio="none" style={{ display: "block" }}>
        {strip.map(({ s, i }: any) => <rect key={i} x={i} y={0} width={1.02} height={1} fill={SPLIT_COLOR[s] || "#333"} />)}
      </svg>
      <div className="row muted" style={{ marginTop: 4 }}>{Object.entries(SPLIT_COLOR).map(([s, c]) => <span key={s} style={{ color: c }}>■ {s}</span>)}
        <span className="spacer" />{Object.entries(meta.split_counts).map(([s, n]: any) => <span key={s}>{s}: {n} commands</span>)}</div>
      {msg && <div className="err">{msg}</div>}
    </div>
    <div className="grid2">
      <div className="card"><div className="row" style={{ marginBottom: 6 }}><Legend /></div>
        {snap && <CyGraph nodes={snap.nodes} edges={snap.edges} showStatic={showStatic} onSelect={setSel} height={560} />}</div>
      <div style={{ display: "grid", gap: 14, alignContent: "start" }}>
        <div className="card"><h3>{sel ? (sel.kind === "node" ? `${sel.data.type} · ${sel.data.label}` : `${sel.data.relation}: ${sel.data.source} → ${sel.data.target}`) : "Selection"}</h3>
          {sel ? <div className="scroll"><FeatureTable features={sel.data.features} /></div> : <div className="muted">Click a node or an edge to see its Table II features{raw ? " (raw units)" : " (normalised on the training split)"}.</div>}</div>
        <div className="card"><h3>Commands scored from this snapshot</h3>
          {snap?.commands_scored_here?.length ? <table><thead><tr><th>time</th><th>task</th><th>tool</th><th>tag</th><th>value</th><th>gateway</th></tr></thead><tbody>
            {snap.commands_scored_here.map((c: any) => <tr key={c.command_idx}><td className="mono">{fmtTime(c.ts)}</td><td className="mono">{c.task_id}</td><td>{c.tool}</td>
              <td className="mono">{c.tag}</td><td className="mono">{c.value}</td><td>{c.decision}</td></tr>)}</tbody></table>
            : <div className="muted">None: no write in [T_k, T_k+1). A command at time t uses the last snapshot with T_k ≤ t (causal rule).</div>}</div>
      </div>
    </div>
  </div>);
}
