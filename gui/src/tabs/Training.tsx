// Training: start / stop / resume (exact, from checkpoints), loss and validation-likelihood
// curves per model (variant × seed), thresholds, device in use.
import { useMemo, useState } from "react";
import { api, fmt, post, usePoll } from "../api";
import { Chart } from "../components/Chart";

const VARIANTS = ["full", "agent_only", "plant_only", "network_only", "no_task"];

export function Training({ runId, system }: { runId: string; system: any }) {
  const [m, setM] = useState<any>(null);
  const [cfg, setCfg] = useState<any>(null);
  const [variants, setVariants] = useState<string[]>(["full"]);
  const [seeds, setSeeds] = useState<string>("");
  const [sel, setSel] = useState<string>("");
  const [msg, setMsg] = useState("");
  usePoll(() => { if (runId) api(`/api/runs/${runId}/models`).then(setM).catch(() => setM({ models: [] })); }, 4000, [runId]);
  usePoll(() => { if (runId) api(`/api/runs/${runId}`).then((r) => setCfg(r)); }, 30000, [runId]);
  const seedList = seeds.trim() ? seeds.split(/[ ,]+/).map(Number).filter((x) => !isNaN(x)) : undefined;
  const start = async (resume: boolean) => {
    try { await post(`/api/runs/${runId}/train`, { variants, seeds: seedList, resume }); setMsg("training started (log in the Experiment tab → Jobs)"); }
    catch (e: any) { setMsg(e.message); }
  };
  const stop = async () => { try { await post(`/api/runs/${runId}/train/stop`); setMsg("stopping after the current chunk; a checkpoint is saved"); } catch (e: any) { setMsg(e.message); } };
  const models = m?.models || [];
  const cur = models.find((x: any) => `${x.variant}_seed${x.seed}` === sel) || models[0];
  const curves = useMemo(() => {
    if (!cur?.history?.length) return null;
    const h = cur.history;
    return { ts: h.map((r: any) => r.epoch), series: [
      { label: "train NLL", color: "#3fb6c4", values: h.map((r: any) => r.train_nll) },
      { label: "validation NLL", color: "#f2a93b", values: h.map((r: any) => r.val_nll) }] };
  }, [cur]);
  const dev = system?.compute;
  return (<div style={{ display: "grid", gap: 14 }}>
    <div className="grid2">
      <div className="card"><h3>Train (paper §IV-B)</h3>
        <div className="row" style={{ marginBottom: 8 }}>{VARIANTS.map((v) => <label key={v} className="row">
          <input type="checkbox" checked={variants.includes(v)} onChange={(e) => setVariants(e.target.checked ? [...variants, v] : variants.filter((x) => x !== v))} />{v}</label>)}</div>
        <div className="row" style={{ marginBottom: 8 }}><span className="muted">seeds</span>
          <input value={seeds} onChange={(e) => setSeeds(e.target.value)} placeholder="config seeds" className="mono" style={{ width: 160 }} />
          <button className="primary" disabled={m?.training || !variants.length} onClick={() => start(true)}>Train / resume</button>
          <button disabled={m?.training || !variants.length} onClick={() => { if (confirm("Delete the selected checkpoints and train from scratch?")) start(false); }}>Restart from scratch</button>
          <button className="danger" disabled={!m?.training} onClick={stop}>Stop (checkpoint)</button></div>
        <div className="muted">“Train / resume” continues each model exactly from its last checkpoint (model, optimiser, GRU memory, RNG, position in the epoch). Hyperparameters come from the run's config.</div>
        {msg && <div className="muted" style={{ marginTop: 6 }}>{msg}</div>}
      </div>
      <div className="card"><h3>Compute</h3>
        <div className="mono">torch {dev?.torch ?? "–"} · CUDA {dev?.cuda ? `yes (${dev.gpus.join(", ")})` : "no"}</div>
        <div className="muted" style={{ marginTop: 6 }}>Run config: device = {cfg?.meta?.device_requested ?? "–"} → {cfg?.meta?.device_resolved ?? "–"} · seeds {cfg?.meta?.seeds?.join(", ") ?? "–"}</div>
        <div className="muted">Defaults (paper): Q = 8, d = 64, 2 HGT layers × 4 heads, Adam 1e-3, 32 snapshots per chunk, early stopping on validation likelihood, threshold at 1 % FPR on benign validation.</div>
      </div>
    </div>
    <div className="card"><h3>Models</h3>
      {models.length === 0 ? <div className="muted">No model trained for this run yet.</div> : <table>
        <thead><tr><th>model</th><th>epochs</th><th>steps</th><th>best epoch</th><th>best val NLL</th><th>threshold (1 % FPR)</th><th>val commands</th><th>params</th><th>status</th></tr></thead><tbody>
          {models.map((x: any) => { const id = `${x.variant}_seed${x.seed}`; const s = x.summary; const last = x.history[x.history.length - 1];
            return <tr key={id} onClick={() => setSel(id)} style={{ cursor: "pointer", background: (cur && id === `${cur.variant}_seed${cur.seed}`) ? "var(--panel2)" : undefined }}>
              <td className="mono">{id}</td><td>{s?.epochs ?? (last ? last.epoch + 1 : 0)}</td><td>{s?.steps ?? last?.step ?? 0}</td><td>{s?.best_epoch ?? "–"}</td>
              <td className="mono">{fmt(s?.best_val_nll, 4)}</td><td className="mono">{fmt(x.threshold?.threshold, 3)}</td><td>{x.threshold?.n_val ?? "–"}</td>
              <td>{s?.params?.toLocaleString() ?? "–"}</td>
              <td>{s?.done ? <span className="badge ok">{s.stop_reason}</span> : x.has_checkpoint ? <span className="badge warn">checkpoint</span> : <span className="badge">–</span>}</td></tr>; })}
        </tbody></table>}
    </div>
    {curves && <div className="card"><h3>{cur.variant} · seed {cur.seed}: negative log-likelihood per epoch</h3>
      <Chart ts={curves.ts} series={curves.series} xTime={false} xLabel="epoch" height={220} /></div>}
  </div>);
}
