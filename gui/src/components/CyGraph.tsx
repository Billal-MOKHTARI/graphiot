// Cytoscape view of one snapshot. Layered, deterministic layout (task -> agent -> gateway
// -> hosts/PLCs -> tags by stage -> stages) so a node keeps its place along the timeline.
// Node type = colour + shape (identity never by colour alone); edge width = attention if given.
import { useEffect, useRef } from "react";
import cytoscape from "cytoscape";

export const NODE_STYLE: Record<string, { color: string; shape: string }> = {
  task: { color: "#3987e5", shape: "round-rectangle" },
  agent: { color: "#d95926", shape: "ellipse" },
  gateway: { color: "#199e70", shape: "diamond" },
  host: { color: "#c98500", shape: "hexagon" },
  plc: { color: "#d55181", shape: "rectangle" },
  tag: { color: "#9085e9", shape: "ellipse" },
  stage: { color: "#e66767", shape: "round-octagon" },
};
const COLUMN: Record<string, number> = { task: 0, agent: 1, gateway: 2, host: 3, plc: 3, tag: 4, stage: 7 };
const STAGE_COL: Record<string, number> = { mix: 0, coat: 1, slit: 2 };

export type GNode = { id: string; type: string; label: string; features?: Record<string, number | null> };
export type GEdge = { id: string; source: string; target: string; relation: string; features?: Record<string, number | null>; weight?: number };

function position(n: GNode, index: Record<string, number>) {
  const i = (index[n.type] = (index[n.type] ?? -1) + 1);
  if (n.type === "tag") {
    const st = n.label.split(".")[0];
    const k = (index[`tag:${st}`] = (index[`tag:${st}`] ?? -1) + 1);
    return { x: 160 * (COLUMN.tag + (STAGE_COL[st] ?? 0)), y: 26 * k };
  }
  if (n.type === "plc") return { x: 160 * COLUMN.plc, y: 60 * i + 500 };
  if (n.type === "stage") return { x: 160 * COLUMN.stage, y: 120 * i + 200 };
  return { x: 160 * (COLUMN[n.type] ?? 0), y: 60 * i + 120 };
}

export function CyGraph({ nodes, edges, onSelect, height = 520, showStatic = false, focus }: {
  nodes: GNode[]; edges: GEdge[]; onSelect?: (sel: { kind: "node" | "edge"; data: any }) => void;
  height?: number; showStatic?: boolean; focus?: string[];
}) {
  const el = useRef<HTMLDivElement>(null);
  const cy = useRef<cytoscape.Core | null>(null);
  const sel = useRef(onSelect);
  sel.current = onSelect;
  useEffect(() => {
    if (!el.current) return;
    const idx: Record<string, number> = {};
    const keep = edges.filter((e) => showStatic || (e.relation !== "controls" && e.relation !== "belongs"));
    const maxW = Math.max(1e-9, ...keep.map((e) => e.weight ?? 0));
    const focusSet = new Set(focus || []);
    const elements: cytoscape.ElementDefinition[] = [
      ...nodes.map((n) => ({ group: "nodes" as const, data: { ...n, color: NODE_STYLE[n.type]?.color, shape: NODE_STYLE[n.type]?.shape,
        focus: focusSet.has(n.id) ? 1 : 0 }, position: position(n, idx) })),
      ...keep.map((e) => ({ group: "edges" as const, data: { ...e, w: e.weight === undefined ? 1 : 0.6 + 5 * (e.weight / maxW) } })),
    ];
    cy.current?.destroy();
    cy.current = cytoscape({
      container: el.current, elements, layout: { name: "preset" }, wheelSensitivity: 0.3,
      style: [
        { selector: "node", style: { "background-color": "data(color)", shape: "data(shape)" as any, width: 16, height: 16,
          label: "data(label)", color: "#c3c2b7", "font-size": 9, "text-valign": "center", "text-halign": "right", "text-margin-x": 4 } },
        { selector: "node[focus = 1]", style: { "border-width": 3, "border-color": "#f0efec", width: 22, height: 22, color: "#f0efec" } },
        { selector: "edge", style: { width: "data(w)", "line-color": "#4a5566", "target-arrow-color": "#4a5566",
          "target-arrow-shape": "triangle", "arrow-scale": 0.6, "curve-style": "bezier", opacity: 0.85 } },
        { selector: "edge[relation = 'writes']", style: { "line-color": "#d95926", "target-arrow-color": "#d95926" } },
        { selector: "edge[relation = 'invokes'], edge[relation = 'assigned']", style: { "line-color": "#3987e5", "target-arrow-color": "#3987e5" } },
        { selector: ":selected", style: { "border-width": 3, "border-color": "#3fb6c4", "line-color": "#3fb6c4", "target-arrow-color": "#3fb6c4" } },
      ],
    });
    cy.current.on("tap", "node", (ev) => sel.current?.({ kind: "node", data: ev.target.data() }));
    cy.current.on("tap", "edge", (ev) => sel.current?.({ kind: "edge", data: ev.target.data() }));
    cy.current.fit(undefined, 20);
    return () => { cy.current?.destroy(); cy.current = null; };
  }, [nodes, edges, showStatic, focus]);
  return <div ref={el} style={{ width: "100%", height, background: "#0a0e13", border: "1px solid var(--line)", borderRadius: 6 }} />;
}

export function Legend() {
  return <div className="row" style={{ gap: 12 }}>{Object.entries(NODE_STYLE).map(([t, s]) =>
    <span key={t} className="row" style={{ gap: 4 }}><svg width="12" height="12"><rect width="12" height="12" rx={s.shape.includes("ellipse") ? 6 : 2} fill={s.color} /></svg>
      <span className="muted">{t}</span></span>)}</div>;
}

export function FeatureTable({ features }: { features?: Record<string, number | null> }) {
  if (!features) return null;
  const rows = Object.entries(features).filter(([k, v]) => !(k.startsWith("type=") || k.startsWith("kind=") || k.startsWith("role=")
    || k.startsWith("phase=") || k.startsWith("stage=") || k.startsWith("tool=") || k.startsWith("FC") || k.startsWith("requests ")) || v);
  return <table><tbody>{rows.map(([k, v]) => <tr key={k}><td>{k}</td><td className="mono">{v === null ? "–" : Number(v).toPrecision(4)}</td></tr>)}</tbody></table>;
}
