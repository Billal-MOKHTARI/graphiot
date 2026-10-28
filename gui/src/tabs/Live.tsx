// Live testbed: 1 Hz tags (as the HMI reads them), phases, outputs vs limits, gateway log.
import { useEffect, useMemo, useRef, useState } from "react";
import { api, fmt, fmtTime, usePoll } from "../api";
import { Chart } from "../components/Chart";

const WINDOW_S = 900;
const STAGES = [{ p: "mix", name: "Mixing" }, { p: "coat", name: "Coating + drying" }, { p: "slit", name: "Slitting" }];
const PHASE = ["idle", "ramp", "steady", "transition"];

export function Live({ runId }: { runId: string }) {
  const [plant, setPlant] = useState<any>(null);
  const series = useRef<Record<string, { t: number[]; v: number[] }>>({});
  const offsets = useRef({ tags: 0, gateway: 0 });
  const [tick, setTick] = useState(0);
  const [calls, setCalls] = useState<any[]>([]);

  useEffect(() => { api("/api/plant").then(setPlant); }, []);
  useEffect(() => { series.current = {}; offsets.current = { tags: 0, gateway: 0 }; setCalls([]); }, [runId]);

  usePoll(async () => {
    if (!runId) return;
    const r = await api(`/api/runs/${runId}/live/tags?offset=${offsets.current.tags}&limit=6000`);
    offsets.current.tags = r.offset;
    for (const rec of r.rows) for (const [tag, val] of Object.entries(rec.values)) {
      const s = (series.current[tag] ||= { t: [], v: [] });
      s.t.push(rec.ts); s.v.push(val as number);
    }
    const now = Math.max(0, ...Object.values(series.current).map((s) => s.t[s.t.length - 1] || 0));
    for (const s of Object.values(series.current)) {
      let i = 0; while (i < s.t.length && s.t[i] < now - WINDOW_S) i++;
      if (i) { s.t.splice(0, i); s.v.splice(0, i); }
    }
    const g = await api(`/api/runs/${runId}/live/gateway?offset=${offsets.current.gateway}&limit=500`);
    offsets.current.gateway = g.offset;
    if (g.rows.length) setCalls((c) => [...c, ...g.rows.filter((x: any) => x.tool)].slice(-200));
    setTick((t) => t + 1);
  }, 2000, [runId]);

  const latest = (tag: string) => { const s = series.current[tag]; return s?.v[s.v.length - 1]; };
  const limited = useMemo(() => plant ? Object.keys(plant.limits) : [], [plant]);
  if (!runId) return <div className="placeholder">Select a run.</div>;
  if (!Object.keys(series.current).length) return <div className="card placeholder">No tag data yet. Bring the testbed up — the HMI writes the 1 Hz tag stream as soon as it runs.</div>;

  return (<div style={{ display: "grid", gap: 14 }}>
    {STAGES.map(({ p, name }) => {
      const ph = latest(`${p}.phase`);
      const outs = limited.filter((t) => t.startsWith(p + "."));
      return (<div className="card" key={p}>
        <div className="row" style={{ marginBottom: 8 }}>
          <h3 style={{ margin: 0 }}>{name}</h3>
          <span className={`badge ${ph === 2 ? "ok" : ph === 1 || ph === 3 ? "warn" : ""}`}>{PHASE[ph ?? 0] ?? "?"}</span>
          <span className="muted">{fmt(latest(`${p}.time_in_phase`), 0)} s in phase · work order {fmt(latest(`${p}.work_order`), 0)}</span>
          {latest(`${p}.alarm`) ? <span className="badge bad">ALARM</span> : null}
        </div>
        <div className="grid3">
          {outs.map((tag) => {
            const s = series.current[tag];
            if (!s) return null;
            const lim = plant.limits[tag];
            return <div key={tag + tick}><Chart ts={s.t} title={`${tag} [${lim.unit}]`} height={140}
              series={[{ label: tag, color: "#3fb6c4", values: s.v }]} limits={{ low: lim.low, high: lim.high }} /></div>;
          })}
        </div>
        <details style={{ marginTop: 8 }}><summary className="muted">All {name.toLowerCase()} tags (current values)</summary>
          <table><thead><tr><th>tag</th><th>kind</th><th>value</th><th>unit</th><th>address</th></tr></thead><tbody>
            {plant.tags.filter((t: any) => t.name.startsWith(p + ".")).map((t: any) =>
              <tr key={t.name}><td className="mono">{t.name}</td><td>{t.kind}</td><td className="mono">{fmt(latest(t.name))}</td><td>{t.unit}</td><td className="mono">{t.iec} · {t.table}:{t.address}</td></tr>)}
          </tbody></table></details>
      </div>);
    })}
    <div className="card">
      <h3>Gateway log — tool calls and allow-list decisions</h3>
      <div className="scroll"><table><thead><tr><th>time</th><th>task</th><th>tool</th><th>tag / stage</th><th>decision</th><th>Modbus</th><th>latency</th></tr></thead><tbody>
        {[...calls].reverse().map((c, i) => <tr key={i}>
          <td className="mono">{fmtTime(c.ts)}</td><td className="mono">{c.task_id}</td><td>{c.tool}</td>
          <td className="mono">{c.args?.name || c.args?.stage || c.args?.product || ""}{c.args?.value !== undefined ? ` = ${c.args.value}` : ""}</td>
          <td><span className={`badge ${c.decision === "allowed" ? "ok" : "bad"}`}>{c.decision}</span> {c.decision !== "allowed" && <span className="muted">{c.reason}</span>}</td>
          <td className="mono">{(c.modbus || []).map((m: any) => `FC${m.fc}@${m.address}`).join(" ")}</td>
          <td className="mono">{fmt(c.latency_ms, 1)} ms</td></tr>)}
      </tbody></table></div>
    </div>
  </div>);
}
