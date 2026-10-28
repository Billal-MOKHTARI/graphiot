// Agents: tasks, operator vs red-team periods, LLM transcripts (ground truth — not seen by the monitor).
import { useEffect, useRef, useState } from "react";
import { api, fmt, fmtTime, usePoll } from "../api";

export function Agents({ runId, events }: { runId: string; events: any[] }) {
  const [periods, setPeriods] = useState<any[]>([]);
  const [tasks, setTasks] = useState<Record<string, any>>({});
  const [llm, setLlm] = useState<any[]>([]);
  const [sel, setSel] = useState<string>("");
  const off = useRef({ periods: 0, tasks: 0, llm: 0 });
  useEffect(() => { off.current = { periods: 0, tasks: 0, llm: 0 }; setPeriods([]); setTasks({}); setLlm([]); setSel(""); }, [runId]);
  usePoll(async () => {
    if (!runId) return;
    const [p, t, l] = await Promise.all(["periods", "tasks", "llm"].map((s) =>
      api(`/api/runs/${runId}/live/${s}?offset=${(off.current as any)[s]}&limit=5000`)));
    off.current = { periods: p.offset, tasks: t.offset, llm: l.offset };
    if (p.rows.length) setPeriods((x) => [...x, ...p.rows]);
    if (t.rows.length) setTasks((x) => ({ ...x, ...Object.fromEntries(t.rows.map((r: any) => [r.task_id, r])) }));
    if (l.rows.length) setLlm((x) => [...x, ...l.rows]);
  }, 3000, [runId]);

  const current = events.filter((e) => e.kind === "task_start").slice(-1)[0];
  const t0 = periods[0]?.start_ts, t1 = periods[periods.length - 1]?.end_ts;
  const span = t0 && t1 ? t1 - t0 : 1;
  const transcript = llm.filter((r) => r.task_id === sel);
  const nRt = periods.filter((p) => p.actor === "red_team").length;
  return (<div style={{ display: "grid", gap: 14 }}>
    <div className="card">
      <h3>Who operated the cell</h3>
      <div className="row" style={{ marginBottom: 6 }}>
        <span className="badge op">operator</span><span className="badge rt">red team</span>
        <span className="muted">{periods.length} tasks · {nRt} red-team episodes · same agent identity and gateway session for both</span>
        {current && <span className="badge warn">now: {current.actor} · {current.task_id}</span>}
      </div>
      <div className="timeline">
        {periods.map((p) => <div key={p.task_id} className="seg" title={`${p.task_id} ${p.actor} ${fmtTime(p.start_ts)}–${fmtTime(p.end_ts)}`}
          onClick={() => setSel(p.task_id)}
          style={{ left: `${((p.start_ts - t0) / span) * 100}%`, width: `${Math.max(0.3, ((p.end_ts - p.start_ts) / span) * 100)}%`,
                   background: p.actor === "red_team" ? "var(--red)" : "var(--op)" }} />)}
      </div>
    </div>
    <div className="grid2">
      <div className="card"><h3>Tasks</h3><div className="scroll" style={{ maxHeight: 520 }}><table>
        <thead><tr><th>task</th><th>actor</th><th>type</th><th>kind</th><th>ended</th><th>start</th></tr></thead><tbody>
        {periods.map((p) => { const t = tasks[p.task_id] || {}; return (
          <tr key={p.task_id} onClick={() => setSel(p.task_id)} style={{ cursor: "pointer", background: sel === p.task_id ? "var(--panel2)" : undefined }}>
            <td className="mono">{p.task_id}{p.episode_id ? ` · ${p.episode_id}` : ""}</td>
            <td><span className={`badge ${p.actor === "red_team" ? "rt" : "op"}`}>{p.actor}</span></td>
            <td>{t.task_type}</td><td>{t.kind}{t.is_override ? " (override)" : ""}</td><td>{p.ended}</td><td className="mono">{fmtTime(p.start_ts)}</td></tr>); })}
        </tbody></table></div></div>
      <div className="card"><h3>Transcript {sel && <span className="mono">{sel}</span>}</h3>
        {!sel ? <div className="muted">Select a task.</div> : <div className="scroll" style={{ maxHeight: 520 }}>
          {tasks[sel] && <p className="muted">{tasks[sel].text}</p>}
          {transcript.map((r, i) => {
            if (r.event === "start") return <details key={i}><summary className="muted">system prompt ({r.role}, {r.model})</summary><div className="log">{r.system}</div><div className="log">{r.user}</div></details>;
            if (r.event === "turn") return <div key={i} style={{ margin: "6px 0" }}>
              <span className="badge">{r.served_by}</span> <span className="muted">turn {r.turn} · {r.stop} · {fmt(r.latency_s, 1)} s</span>
              {r.meta?.recovered_text_tool_call && <span className="badge warn">tool call recovered from text</span>}
              {r.meta?.refusal && <span className="badge bad">refusal</span>}
              {r.text && <div className="log">{r.text}</div>}</div>;
            if (r.event === "tool") return <div key={i} className="mono" style={{ fontSize: 12, margin: "2px 0 2px 12px" }}>
              → {r.name}({JSON.stringify(r.args)}) <span className={`badge ${r.ok ? "ok" : "bad"}`}>{r.ok ? "ok" : "error"}</span> <span className="muted">{typeof r.result === "string" ? r.result : JSON.stringify(r.result)?.slice(0, 160)}</span></div>;
            if (r.event === "end") return <div key={i} className="muted">ended: {r.ended} · {r.tool_calls} calls · {r.writes} writes · {r.denied} denied</div>;
            if (r.event === "error") return <div key={i} className="err">{r.error}</div>;
            return null;
          })}</div>}
      </div>
    </div>
  </div>);
}
