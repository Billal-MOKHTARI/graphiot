// Form generated from the experiment config's JSON Schema (pydantic model_json_schema()).
import { useState } from "react";

type Schema = any;
const resolve = (root: Schema, s: Schema): Schema => {
  if (s?.$ref) return resolve(root, root.$defs[s.$ref.split("/").pop()]);
  if (s?.anyOf) { const nn = s.anyOf.filter((x: Schema) => x.type !== "null"); if (nn.length === 1) return { ...resolve(root, nn[0]), nullable: true, description: s.description, title: s.title }; }
  if (s?.allOf?.length === 1) return { ...resolve(root, s.allOf[0]), description: s.description };
  return s;
};

export function SchemaForm({ schema, value, onChange }: { schema: Schema; value: any; onChange: (v: any) => void }) {
  return <Node root={schema} s={schema} v={value} path="" set={onChange} depth={0} />;
}

function Node({ root, s, v, set, path, depth }: { root: Schema; s: Schema; v: any; set: (x: any) => void; path: string; depth: number }) {
  const sch = resolve(root, s);
  const [open, setOpen] = useState(depth < 1);
  if (sch.type === "object" && sch.properties) {
    const body = Object.entries(sch.properties).map(([k, sub]) => (
      <Node key={k} root={root} s={sub} v={v?.[k]} path={path ? `${path}.${k}` : k} depth={depth + 1}
            set={(x) => set({ ...(v || {}), [k]: x })} />
    ));
    if (depth === 0) return <div>{body}</div>;
    return (
      <fieldset>
        <legend onClick={() => setOpen(!open)}>{open ? "▾" : "▸"} {path.split(".").pop()}</legend>
        {open && body}
      </fieldset>
    );
  }
  if (sch.type === "object" && sch.additionalProperties) {   // maps (e.g. task weights, IP maps)
    return <Leaf label={path.split(".").pop()!} hint={sch.description}>
      <textarea className="mono" style={{ width: "100%", minHeight: 60 }} value={JSON.stringify(v ?? {}, null, 0)}
        onChange={(e) => { try { set(JSON.parse(e.target.value)); } catch { /* wait for valid JSON */ } }} />
    </Leaf>;
  }
  const label = path.split(".").pop()!;
  const hint = sch.description;
  if (sch.enum || sch.const !== undefined) {
    const opts = sch.enum ?? [sch.const];
    return <Leaf label={label} hint={hint}>
      <select value={v ?? ""} onChange={(e) => set(typeof opts[0] === "number" ? Number(e.target.value) : e.target.value)}>
        {opts.map((o: any) => <option key={String(o)} value={o}>{String(o)}</option>)}
      </select></Leaf>;
  }
  if (sch.type === "boolean") return <Leaf label={label} hint={hint}><input type="checkbox" checked={!!v} onChange={(e) => set(e.target.checked)} /></Leaf>;
  if (sch.type === "integer" || sch.type === "number") {
    return <Leaf label={label} hint={hint}>
      <input type="number" value={v ?? ""} step={sch.type === "integer" ? 1 : "any"} min={sch.minimum ?? sch.exclusiveMinimum}
             max={sch.maximum ?? sch.exclusiveMaximum}
             onChange={(e) => set(e.target.value === "" ? (sch.nullable ? null : 0) : Number(e.target.value))} /></Leaf>;
  }
  if (sch.type === "array") {
    const item = resolve(root, sch.items || {});
    return <Leaf label={label} hint={hint}>
      <input className="mono" style={{ width: "100%" }} value={(v ?? []).join(", ")}
        onChange={(e) => set(e.target.value.split(",").map((x) => x.trim()).filter(Boolean)
          .map((x) => (item.type === "integer" || item.type === "number" ? Number(x) : x)))} /></Leaf>;
  }
  return <Leaf label={label} hint={hint}><input style={{ width: "100%" }} value={v ?? ""} onChange={(e) => set(e.target.value === "" && sch.nullable ? null : e.target.value)} /></Leaf>;
}

function Leaf({ label, hint, children }: { label: string; hint?: string; children: React.ReactNode }) {
  return <div className="field"><label title={hint}>{label}</label><div>{children}</div>{hint && <div className="hint">{hint}</div>}</div>;
}
