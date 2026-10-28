// Oracle: episode labels (causal + paper-literal), violations, command labels, twin fidelity.
import { useState } from "react";
import { api, fmt, fmtTime, usePoll } from "../api";

export function Oracle({ runId }: { runId: string }) {
  const [o, setO] = useState<any>(null);
  usePoll(() => { if (runId) api(`/api/runs/${runId}/oracle`).then(setO); }, 8000, [runId]);
  if (!o) return <div className="placeholder">Loading…</div>;
  if (!o.available) return <div className="card placeholder">No oracle output yet. Run “Oracle + pilot test” from the Experiment tab after a campaign.</div>;
  const s = o.summary;
  const harmfulCmds = (o.command_labels || []).filter((c: any) => c.harmful);
  return (<div style={{ display: "grid", gap: 14 }}>
    <div className="grid3">
      <div className="card"><h3>Episodes</h3><div className="kpi">{s.episodes} <small>({s.red_team_episodes} red team)</small></div>
        <div className="muted">harmful: {s.harmful_red_team_episodes} red team · {s.harmful_operator_episodes} operator</div></div>
      <div className="card"><h3>Operator commands flagged</h3><div className="kpi">{s.operator_commands_flagged_share == null ? "–" : `${(100 * s.operator_commands_flagged_share).toFixed(1)} %`}</div>
        <div className="muted">of {s.operator_commands} forwarded operator commands</div></div>
      <div className="card"><h3>Twin fidelity</h3>
        <div className="kpi">{s.fidelity_ok == null ? "n/a" : <span className={`badge ${s.fidelity_ok ? "ok" : "bad"}`}>{s.fidelity_ok ? "trusted" : "check"}</span>}</div>
        <div className="muted">worst p95 error {fmt(s.fidelity_worst_p95_rel_err, 3)} of limit width · writes from {s.write_source}</div></div>
    </div>
    <div className="card"><h3>Oracle settings</h3><div className="mono">mode={s.mode} · τ={s.tau} · T_min={s.t_min_s} s · command rule={s.command_rule} · replay {s.replay_seconds} s · violations {s.violations ?? "–"} (unattributed {s.unattributed_violations ?? "–"}) · harmful episodes with no harmful command (Q9): {s.harmful_episodes_with_no_harmful_command}</div></div>
    <div className="card"><h3>Episodes</h3><div className="scroll"><table>
      <thead><tr><th>task</th><th>actor</th><th>technique</th><th>start</th><th>harmful (causal)</th><th>harmful (paper-literal window)</th><th>caused outputs</th><th>writes</th></tr></thead><tbody>
      {o.episode_labels.map((e: any) => <tr key={e.task_id}>
        <td className="mono">{e.task_id}{e.episode_id ? ` · ${e.episode_id}` : ""}</td>
        <td><span className={`badge ${e.actor === "red_team" ? "rt" : "op"}`}>{e.actor}</span></td><td>{e.technique || ""}</td>
        <td className="mono">{fmtTime(e.start_ts)}</td>
        <td><span className={`badge ${e.harmful ? "bad" : "ok"}`}>{e.harmful ? "harmful" : "benign"}</span></td>
        <td>{e.harmful_literal === undefined ? "–" : <span className={`badge ${e.harmful_literal ? "warn" : ""}`}>{e.harmful_literal ? "yes" : "no"}</span>}</td>
        <td className="mono">{(e.caused_outputs || []).join(", ")}</td><td>{e.n_writes}</td></tr>)}
    </tbody></table></div></div>
    <div className="grid2">
      <div className="card"><h3>Violations (twin replay)</h3><div className="scroll"><table>
        <thead><tr><th>output</th><th>start</th><th>duration</th><th>peak (× width)</th><th>attributed to</th></tr></thead><tbody>
        {o.violations.map((v: any, i: number) => <tr key={i}><td className="mono">{v.output}</td><td className="mono">{fmtTime(v.start)}</td>
          <td>{fmt(v.duration_s, 0)} s</td><td>{fmt(v.peak_rel, 3)}</td><td className="mono">{v.attributed_to || "–"}</td></tr>)}
      </tbody></table></div></div>
      <div className="card"><h3>Harmful commands ({harmfulCmds.length})</h3><div className="scroll"><table>
        <thead><tr><th>time</th><th>task</th><th>actor</th><th>tool</th><th>tag</th><th>value</th></tr></thead><tbody>
        {harmfulCmds.map((c: any) => <tr key={c.command_idx}><td className="mono">{fmtTime(c.ts)}</td><td className="mono">{c.task_id}</td>
          <td><span className={`badge ${c.actor === "red_team" ? "rt" : "op"}`}>{c.actor}</span></td><td>{c.tool}</td><td className="mono">{c.tag}</td><td className="mono">{fmt(c.value)}</td></tr>)}
      </tbody></table></div></div>
    </div>
    {o.fidelity?.per_output && <div className="card"><h3>Fidelity per output (twin vs logged tags)</h3><table>
      <thead><tr><th>output</th><th>samples</th><th>mean err</th><th>p95 err</th><th>max err</th></tr></thead><tbody>
      {Object.entries(o.fidelity.per_output).map(([k, v]: any) => <tr key={k}><td className="mono">{k}</td><td>{v.n}</td>
        <td>{fmt(v.mean_rel_err, 4)}</td><td>{fmt(v.p95_rel_err, 4)}</td><td>{fmt(v.max_rel_err, 4)}</td></tr>)}
    </tbody></table></div>}
  </div>);
}
