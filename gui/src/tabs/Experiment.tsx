import { useEffect, useState } from "react";
import { api, post } from "../api";
import { SchemaForm } from "../components/SchemaForm";
import type { Job } from "../App";

export function Experiment({ runId, setRunId, runs, jobs, refreshRuns, system }: {
  runId: string; setRunId: (s: string) => void; runs: any[]; jobs: Record<string, Job>; refreshRuns: () => void; system: any;
}) {
  const [configs, setConfigs] = useState<any[]>([]);
  const [name, setName] = useState("pilot");
  const [text, setText] = useState("");
  const [schema, setSchema] = useState<any>(null);
  const [check, setCheck] = useState<any>(null);
  const [formView, setFormView] = useState(false);
  const [msg, setMsg] = useState("");
  const [secretName, setSecretName] = useState("GEMINI_API_KEY");
  const [secretVal, setSecretVal] = useState("");

  useEffect(() => { api("/api/configs").then(setConfigs); api("/api/schema").then(setSchema); }, []);
  useEffect(() => { api(`/api/configs/${name}`).then((c) => { setText(c.text); setCheck(null); }).catch(() => {}); }, [name]);

  const validate = async (t = text) => { const r = await post("/api/config/validate", { text: t }); setCheck(r); return r; };
  const onForm = async (v: any) => {                         // JSON is valid YAML: let the backend normalize it
    const r = await validate(JSON.stringify(v));
    if (r.ok) setText(r.yaml);
  };
  const save = async () => {
    const target = prompt("Save config as (name):", name);
    if (!target) return;
    try { await post(`/api/configs/${target}`, { text }); setMsg(`saved configs/${target}.yaml`); setName(target); api("/api/configs").then(setConfigs); }
    catch (e: any) { setCheck(e.body); }
  };
  const newRun = async () => {
    try { const r = await post("/api/runs", { text }); setRunId(r.run_id); refreshRuns(); setMsg(`created run ${r.run_id}`); }
    catch (e: any) { setCheck(e.body); }
  };
  const act = async (path: string) => { try { await post(path); setMsg(""); refreshRuns(); } catch (e: any) { setMsg(e.body?.detail || e.message); } };
  const [pre, setPre] = useState<any>(null);
  useEffect(() => { setPre(null); if (runId) api(`/api/runs/${runId}/preflight`).then(setPre).catch(() => {}); }, [runId]);
  const download = () => { const a = document.createElement("a"); a.href = URL.createObjectURL(new Blob([text], { type: "text/yaml" })); a.download = `${name}.yaml`; a.click(); };
  const upload = (f?: File) => f && f.text().then((t) => { setText(t); validate(t); });

  const run = runs.find((r) => r.run_id === runId);
  const runJobs = Object.values(jobs).filter((j) => j.run_id === runId).slice(0, 6);
  return (
    <div className="grid2">
      <div className="card">
        <h3>Experiment configuration</h3>
        <div className="row" style={{ marginBottom: 8 }}>
          <select value={name} onChange={(e) => setName(e.target.value)}>{configs.map((c) => <option key={c.name}>{c.name}</option>)}</select>
          <button onClick={() => validate()}>Validate</button>
          <button onClick={() => setFormView(!formView)} disabled={!check?.ok && formView === false && !check}>
            {formView ? "YAML view" : "Form view"}</button>
          <button onClick={save}>Save as…</button>
          <button onClick={download}>Download</button>
          <label className="row"><span className="muted">Upload</span><input type="file" accept=".yaml,.yml" onChange={(e) => upload(e.target.files?.[0])} /></label>
          <span className="spacer" />
          <button className="primary" onClick={newRun}>Create run from this config</button>
        </div>
        {check && (check.ok ? <div className="row"><span className="badge ok">valid</span><span className="muted mono">sha256 {check.hash.slice(0, 16)}…</span></div>
          : <div className="err">{(check.errors || []).map((e: any) => `${e.loc}: ${e.msg}`).join("\n")}</div>)}
        {msg && <div className="muted" style={{ margin: "6px 0" }}>{msg}</div>}
        {formView && schema && check?.ok
          ? <div style={{ maxHeight: 560, overflow: "auto", marginTop: 8 }}><SchemaForm schema={schema} value={check.normalized} onChange={onForm} /></div>
          : <textarea className="code" value={text} onChange={(e) => setText(e.target.value)} onBlur={() => validate()} spellCheck={false} />}
        {formView && !check?.ok && <div className="muted">Validate the YAML first to open the form.</div>}
      </div>
      <div style={{ display: "grid", gap: 14, alignContent: "start" }}>
        <div className="card">
          <h3>Run</h3>
          {!run ? <div className="muted">No run selected. Create one from a config.</div> : <>
            <div className="mono" style={{ marginBottom: 8 }}>{run.run_id}</div>
            <div className="row" style={{ marginBottom: 8 }}>
              <span className={`badge ${run.testbed_up ? "ok" : run.testbed_running ? "warn" : ""}`}>
                testbed {run.testbed_up ? "up" : run.testbed_running ? "running (not started here)" : "down"}</span>
              <span className={`badge ${run.campaign_running ? "warn" : ""}`}>campaign {run.campaign_running ? "running" : "idle"}</span>
              <span className={`badge ${run.has_oracle ? "ok" : ""}`}>oracle {run.has_oracle ? "done" : "–"}</span>
              <span className={`badge ${run.has_pilot ? "ok" : ""}`}>pilot {run.has_pilot ? "done" : "–"}</span>
            </div>
            <div className="row">
              <button className="primary" disabled={run.testbed_up || !system?.docker} onClick={() => act(`/api/runs/${runId}/testbed/up`)}>Testbed up</button>
              <button disabled={!run.testbed_up || run.campaign_running} onClick={() => act(`/api/runs/${runId}/campaign/start`)}>Start campaign</button>
              <button className="danger" disabled={!run.campaign_running} onClick={() => act(`/api/runs/${runId}/campaign/stop`)}>Stop campaign</button>
              <button className="danger" disabled={!run.testbed_up && !run.testbed_running} onClick={() => act(`/api/runs/${runId}/testbed/down`)}>Testbed down</button>
              <button disabled={run.campaign_running} onClick={() => act(`/api/runs/${runId}/process`)}>Oracle + pilot test</button>
            </div>
            {pre && !pre.ok && <div className="err" style={{ marginTop: 8 }}>LLM preflight: {pre.problems.join("; ")}</div>}
            {pre?.ok && <div className="muted" style={{ marginTop: 8 }}>LLM preflight: both agents' models are available.</div>}
            {runs.some((r) => r.testbed_running && r.run_id !== runId) &&
              <div className="muted" style={{ marginTop: 8 }}>Another run's testbed is running ({runs.filter((r) => r.testbed_running && r.run_id !== runId).map((r) => r.run_id).join(", ")}); only one testbed can run at a time.</div>}
            </>}
        </div>
        <div className="card">
          <h3>Jobs</h3>
          {runJobs.length === 0 && <div className="muted">No jobs for this run yet.</div>}
          {runJobs.map((j) => (
            <div key={j.id} style={{ marginBottom: 10 }}>
              <div className="row"><b>{j.kind}</b><span className={`badge ${j.status === "done" ? "ok" : j.status === "error" ? "bad" : "warn"}`}>{j.status}</span>
                {j.error && <span className="err">{j.error}</span>}</div>
              <div className="log">{j.log.join("\n") || "…"}</div>
            </div>))}
        </div>
        <div className="card">
          <h3>API keys (memory only)</h3>
          <div className="muted" style={{ marginBottom: 6 }}>Keys stay in the backend's memory (or your .env). They are never written to YAML, logs or exports. Local Ollama needs none.</div>
          <div className="row">
            <input value={secretName} onChange={(e) => setSecretName(e.target.value)} placeholder="ENV_VAR_NAME" className="mono" />
            <input type="password" value={secretVal} onChange={(e) => setSecretVal(e.target.value)} placeholder="value" />
            <button onClick={async () => { await post("/api/secrets", { name: secretName, value: secretVal }); setSecretVal(""); setMsg(`${secretName} set (in memory)`); }}>Set</button>
          </div>
          <div className="muted" style={{ marginTop: 6 }}>Available: {(system?.secrets || []).join(", ") || "none"}</div>
        </div>
      </div>
    </div>
  );
}
