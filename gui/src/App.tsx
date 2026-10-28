import { useEffect, useState } from "react";
import { api, usePoll, useSocket } from "./api";
import { Experiment } from "./tabs/Experiment";
import { Live } from "./tabs/Live";
import { Agents } from "./tabs/Agents";
import { Oracle } from "./tabs/Oracle";
import { Results } from "./tabs/Results";
import { Graph } from "./tabs/Graph";
import { Training } from "./tabs/Training";
import { Inference } from "./tabs/Inference";

const TABS = ["Experiment", "Live testbed", "Agents", "Graph", "Training", "Inference & explainability", "Oracle", "Results"] as const;
type Tab = (typeof TABS)[number];

export type Job = { id: string; kind: string; run_id: string | null; status: string; log: string[]; error?: string };

export function App() {
  // ?tab=…&run=… open a given view (shareable links); otherwise the last one used.
  const q = new URLSearchParams(location.search);
  const [tab, setTab] = useState<Tab>(() => (TABS.includes(q.get("tab") as Tab) ? q.get("tab") as Tab : (localStorage.getItem("tab") as Tab) || "Experiment"));
  const [runs, setRuns] = useState<any[]>([]);
  const [runId, setRunId] = useState<string>(() => q.get("run") || localStorage.getItem("run") || "");
  const [jobs, setJobs] = useState<Record<string, Job>>({});
  const [system, setSystem] = useState<any>(null);
  const [events, setEvents] = useState<any[]>([]);

  useEffect(() => { try { localStorage.setItem("tab", tab); } catch { /* private mode */ } }, [tab]);
  useEffect(() => { try { localStorage.setItem("run", runId); } catch { /* private mode */ } }, [runId]);
  const refreshRuns = () => api("/api/runs").then((r) => { setRuns(r); if (!runId && r[0]) setRunId(r[0].run_id); });
  usePoll(refreshRuns, 5000, [runId]);
  usePoll(() => api("/api/system").then(setSystem), 15000);
  usePoll(() => api<Job[]>("/api/jobs").then((js) => setJobs(Object.fromEntries(js.map((j) => [j.id, j])))), 10000);

  useSocket((m) => {
    if (m.type === "job") setJobs((j) => ({ ...j, [m.job.id]: m.job }));
    if (m.type === "job_log") setJobs((j) => j[m.job_id] ? { ...j, [m.job_id]: { ...j[m.job_id], log: [...j[m.job_id].log, m.line].slice(-300) } } : j);
    if (m.type === "campaign") setEvents((e) => [...e, m.event].slice(-400));
    if (m.type === "testbed") refreshRuns();
  });

  const running = Object.values(jobs).filter((j) => j.status === "running");
  const run = runs.find((r) => r.run_id === runId);
  return (
    <div className="app">
      <div className="topbar">
        <span className="brand">GRAPHIOT<small>task-aware GNN · harmful LLM-agent commands in IIoT</small></span>
        <span className="spacer" />
        <span className="muted">run</span>
        <select value={runId} onChange={(e) => setRunId(e.target.value)} style={{ maxWidth: 360 }}>
          {runs.map((r) => <option key={r.run_id} value={r.run_id}>{r.run_id}{r.testbed_up ? "  ● testbed up" : ""}</option>)}
        </select>
        {run && <span className={`badge ${run.for_paper ? "ok" : "warn"}`}>{run.for_paper ? "paper run" : "not for paper"}</span>}
        {run?.testbed_backend === "dev" && <span className="badge bad">DEV BACKEND</span>}
        <span className={`badge ${system?.docker ? "ok" : "bad"}`}>docker {system?.docker ? "ok" : "unavailable"}</span>
        <span className="badge">{running.length} job{running.length === 1 ? "" : "s"} running</span>
      </div>
      <div className="tabs">
        {TABS.map((t) => <div key={t} className={`tab ${t === tab ? "active" : ""}`} onClick={() => setTab(t)}>{t}</div>)}
      </div>
      <div className="content">
        {tab === "Experiment" && <Experiment runId={runId} setRunId={setRunId} runs={runs} jobs={jobs} refreshRuns={refreshRuns} system={system} />}
        {tab === "Live testbed" && <Live runId={runId} />}
        {tab === "Agents" && <Agents runId={runId} events={events} />}
        {tab === "Graph" && <Graph runId={runId} />}
        {tab === "Training" && <Training runId={runId} system={system} />}
        {tab === "Inference & explainability" && <Inference runId={runId} />}
        {tab === "Oracle" && <Oracle runId={runId} />}
        {tab === "Results" && <Results runId={runId} />}
      </div>
    </div>
  );
}
