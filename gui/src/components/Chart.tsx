// uPlot wrapper: time series with optional horizontal limit lines.
import { useEffect, useRef } from "react";
import uPlot from "uplot";
import "uplot/dist/uPlot.min.css";

type Series = { label: string; color: string; values: (number | null)[] };
export function Chart({ ts, series, limits, height = 160, title, xTime = true, xLabel }: {
  ts: number[]; series: Series[]; limits?: { low?: number | null; high?: number | null }; height?: number; title?: string;
  xTime?: boolean; xLabel?: string;
}) {
  const el = useRef<HTMLDivElement>(null);
  const plot = useRef<uPlot | null>(null);
  useEffect(() => {
    if (!el.current) return;
    const extra: Series[] = [];
    if (limits?.high != null) extra.push({ label: "high limit", color: "#f0544f", values: ts.map(() => limits.high!) });
    if (limits?.low != null) extra.push({ label: "low limit", color: "#f2a93b", values: ts.map(() => limits.low!) });
    const all = [...series, ...extra];
    const opts: uPlot.Options = {
      title, width: el.current.clientWidth || 400, height,
      scales: { x: { time: xTime } },
      axes: [{ stroke: "#8b97a8", grid: { stroke: "#232b36" }, label: xLabel }, { stroke: "#8b97a8", grid: { stroke: "#232b36" }, size: 50 }],
      series: [{}, ...all.map((s, i) => ({ label: s.label, stroke: s.color, width: i >= series.length ? 1 : 1.5,
                                            dash: i >= series.length ? [6, 4] : undefined, spanGaps: true }))],
      legend: { show: true },
    };
    const data = [ts, ...all.map((s) => s.values)] as uPlot.AlignedData;
    plot.current?.destroy();
    plot.current = new uPlot(opts, data, el.current);
    const ro = new ResizeObserver(() => plot.current?.setSize({ width: el.current!.clientWidth, height }));
    ro.observe(el.current);
    return () => { ro.disconnect(); plot.current?.destroy(); plot.current = null; };
  }, [ts, series, limits, height, title, xTime, xLabel]);
  return <div ref={el} className="chart" />;
}
