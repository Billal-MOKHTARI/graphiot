// Thin client for the FastAPI backend + a reconnecting WebSocket hook.
import { useEffect, useRef } from "react";

export async function api<T = any>(path: string, opts: RequestInit = {}): Promise<T> {
  const r = await fetch(path, { headers: { "Content-Type": "application/json" }, ...opts });
  const text = await r.text();
  const body = text ? JSON.parse(text) : null;
  if (!r.ok) throw Object.assign(new Error(body?.detail || r.statusText), { status: r.status, body });
  return body as T;
}
export const post = <T = any>(path: string, data?: unknown) =>
  api<T>(path, { method: "POST", body: data === undefined ? undefined : JSON.stringify(data) });

export function useSocket(onMessage: (m: any) => void) {
  const cb = useRef(onMessage);
  cb.current = onMessage;
  useEffect(() => {
    let ws: WebSocket | null = null, timer: number | undefined, alive = true;
    const connect = () => {
      const proto = location.protocol === "https:" ? "wss" : "ws";
      ws = new WebSocket(`${proto}://${location.host}/api/ws`);
      ws.onmessage = (e) => { try { cb.current(JSON.parse(e.data)); } catch { /* ignore */ } };
      ws.onclose = () => { if (alive) timer = window.setTimeout(connect, 2000); };
    };
    connect();
    const ping = window.setInterval(() => ws?.readyState === 1 && ws.send("ping"), 20000);
    return () => { alive = false; clearTimeout(timer); clearInterval(ping); ws?.close(); };
  }, []);
}

export function usePoll(fn: () => void, ms: number, deps: unknown[] = []) {
  useEffect(() => {
    fn();
    const id = window.setInterval(fn, ms);
    return () => clearInterval(id);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps);
}

export const fmtTime = (ts?: number | null) =>
  ts ? new Date(ts * 1000).toLocaleTimeString([], { hour12: false }) : "–";
export const fmt = (v: any, d = 2) =>
  v === null || v === undefined ? "–" : typeof v === "number" ? (Number.isInteger(v) ? String(v) : v.toFixed(d)) : String(v);
