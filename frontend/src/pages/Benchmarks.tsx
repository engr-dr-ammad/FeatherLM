import { useEffect, useState } from "react";
import { Bar, BarChart, CartesianGrid, Legend, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { fmt, get } from "../api";

type Row = Record<string, number | string | boolean | null>;
interface Results { [name: string]: { rows?: Row[]; models?: Row[]; system?: Row; summary?: Row[]; onnx?: Row; [k: string]: unknown } }

const COLORS = ["#1c1917", "#d97706", "#0f766e", "#7c3aed", "#be123c"];

function Table({ rows, cols }: { rows: Row[]; cols?: string[] }) {
  if (!rows.length) return null;
  const keys = cols ?? Object.keys(rows[0]).filter((k) => typeof rows[0][k] !== "object");
  return (
    <div className="overflow-x-auto">
      <table className="table">
        <thead><tr>{keys.map((k) => <th key={k}>{k.replace(/_/g, " ")}</th>)}</tr></thead>
        <tbody>{rows.map((r, i) => <tr key={i}>{keys.map((k) => <td key={k} className="font-mono text-xs">{show(r[k])}</td>)}</tr>)}</tbody>
      </table>
    </div>
  );
}
const show = (v: unknown) => (typeof v === "number" ? (Number.isInteger(v) ? v.toLocaleString() : v.toFixed(4)) : String(v ?? "—"));

function Bars({ data, x, keys, title, unit }: { data: Row[]; x: string; keys: string[]; title: string; unit?: string }) {
  return (
    <div className="card">
      <h3 className="mb-2 text-sm font-semibold">{title}</h3>
      <div className="h-60">
        <ResponsiveContainer>
          <BarChart data={data}>
            <CartesianGrid stroke="#e7e5e4" strokeDasharray="3 3" vertical={false} />
            <XAxis dataKey={x} tick={{ fontSize: 11 }} />
            <YAxis tick={{ fontSize: 11 }} unit={unit} />
            <Tooltip />
            {keys.length > 1 && <Legend />}
            {keys.map((k, i) => <Bar key={k} dataKey={k} fill={COLORS[i % COLORS.length]} radius={[3, 3, 0, 0]} />)}
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}

function pivot(rows: Row[], x: string, series: string, y: string) {
  const out = new Map<number | string, Row>();
  for (const r of rows) {
    const k = r[x] as number;
    const row = out.get(k) ?? { [x]: k };
    row[String(r[series])] = r[y];
    out.set(k, row);
  }
  return [...out.values()].sort((a, b) => (a[x] as number) - (b[x] as number));
}

function Lines({ rows, x, series, y, title, unit }: { rows: Row[]; x: string; series: string; y: string; title: string; unit?: string }) {
  const data = pivot(rows, x, series, y);
  const names = [...new Set(rows.map((r) => String(r[series])))];
  return (
    <div className="card">
      <h3 className="mb-2 text-sm font-semibold">{title}</h3>
      <div className="h-60">
        <ResponsiveContainer>
          <LineChart data={data}>
            <CartesianGrid stroke="#e7e5e4" strokeDasharray="3 3" />
            <XAxis dataKey={x} tick={{ fontSize: 11 }} />
            <YAxis tick={{ fontSize: 11 }} unit={unit} />
            <Tooltip /><Legend />
            {names.map((n, i) => <Line key={n} dataKey={n} stroke={COLORS[i % COLORS.length]} strokeWidth={2} />)}
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}

export default function Benchmarks() {
  const [res, setRes] = useState<Results | null>(null);
  const [err, setErr] = useState("");
  useEffect(() => { get<{ results: Results }>("/benchmarks").then((r) => setRes(r.results)).catch((e) => setErr(e.message)); }, []);

  if (err) return <div className="card text-sm text-red-600">{err}</div>;
  if (!res) return <div className="card text-sm text-stone-500">Loading…</div>;
  if (!Object.keys(res).length) return <div className="card text-sm text-stone-500">No benchmark results yet. Run <code>scripts/run_benchmarks.sh</code>.</div>;

  const quality = (res.quality_summary?.rows ?? []) as Row[];
  const nano = (res.nano_comparison?.rows ?? []) as Row[];
  const memRows = (res.memory?.rows ?? []) as Row[];
  const trainMem = memRows.filter((r) => r.mode === "train").map((r) => ({ ...r, label: `${r.backend}${r.gradient_checkpointing ? " +ckpt" : ""}` }));
  const inferMem = memRows.filter((r) => r.mode === "infer");
  const speed = (res.speed?.rows ?? []) as Row[];
  const quant = (res.quantization?.models ?? []) as Row[];
  const sys = (res.quality?.system ?? res.memory?.system ?? res.speed?.system) as Row | undefined;

  return (
    <div className="space-y-4">
      <div className="rounded-lg border border-amber-200 bg-amber-50 p-3 text-sm text-amber-900">
        Measured results from this repository's benchmark scripts on the hardware listed below. Small-scale experiments: they do not by themselves establish that one architecture is better.
      </div>
      {nano.length > 0 && (
        <section className="space-y-3">
          <h2 className="font-semibold">Nano: Feather core vs. Transformer baseline (Tiny Shakespeare)</h2>
          <div className="card"><Table rows={nano} /></div>
        </section>
      )}
      {quality.length > 0 && (
        <section className="space-y-3">
          <h2 className="font-semibold">Matched-budget comparison (multiple seeds)</h2>
          <div className="card"><Table rows={quality} /></div>
          <div className="grid gap-4 md:grid-cols-3">
            <Bars data={quality} x="arch" keys={["val_loss_mean"]} title="Validation loss (lower is better)" />
            <Bars data={quality} x="arch" keys={["peak_train_rss_mb_mean"]} title="Peak training RAM" unit=" MB" />
            <Bars data={quality} x="arch" keys={["train_tokens_per_sec_mean"]} title="Training throughput (tok/s)" />
          </div>
        </section>
      )}
      {(trainMem.length > 0 || inferMem.length > 0) && (
        <section className="space-y-3">
          <h2 className="font-semibold">Memory</h2>
          <div className="grid gap-4 md:grid-cols-2">
            {trainMem.length > 0 && <Bars data={trainMem} x="label" keys={["peak_rss_mb"]} title="Peak training RAM (process RSS)" unit=" MB" />}
            {inferMem.length > 0 && <Lines rows={inferMem} x="context" series="backend" y="cache_mb" title="Inference cache size vs. context length" unit=" MB" />}
          </div>
        </section>
      )}
      {speed.length > 0 && (
        <section className="space-y-3">
          <h2 className="font-semibold">CPU inference speed</h2>
          <div className="grid gap-4 md:grid-cols-2">
            <Lines rows={speed} x="prompt_len" series="backend" y="ttft_ms" title="Time to first token vs. prompt length" unit=" ms" />
            <Lines rows={speed} x="prompt_len" series="backend" y="decode_tokens_per_sec" title="Decode throughput vs. context (tok/s)" />
          </div>
        </section>
      )}
      {quant.length > 0 && (
        <section className="space-y-3">
          <h2 className="font-semibold">Quantization</h2>
          <div className="card">
            <Table rows={quant.map((q) => ({ ...q, file: fmt.bytes(q.file_bytes as number), resident: fmt.bytes(q.resident_weight_bytes as number) }))}
              cols={["quantization", "runtime", "val_perplexity", "top1_agreement", "max_abs_logit_diff", "decode_tokens_per_sec", "file", "resident"]} />
          </div>
        </section>
      )}
      {sys && <p className="font-mono text-xs text-stone-500">Hardware: {String(sys.cpu)} · {String(sys.cpu_count_logical)} threads · {String(sys.ram_total_mb)} MB RAM · {String(sys.os)} · Python {String(sys.python)} · torch {String(sys.torch)}</p>}
    </div>
  );
}
