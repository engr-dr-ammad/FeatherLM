import { useEffect, useState } from "react";
import { CartesianGrid, Legend, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { fmt, get, post } from "../api";

interface Job {
  id: string; name: string; base_config: string; output_dir: string; status: string; created: number; error?: string | null;
  progress: {
    state?: string; step?: number; max_steps?: number; train_loss?: number; val_loss?: number; best_val_loss?: number; lr?: number;
    tokens_seen?: number; rss_mb?: number; peak_rss_mb?: number; cpu_percent?: number; tokens_per_sec?: number;
    last_checkpoint?: string; checkpoint_step?: number; dataset?: string; model?: string; backend?: string; params?: number;
  };
}
interface Metric { type: string; step: number; train_loss?: number; val_loss?: number; lr?: number }

const statusColor: Record<string, string> = {
  running: "bg-blue-100 text-blue-800", completed: "bg-emerald-100 text-emerald-800", stopped: "bg-stone-200 text-stone-700",
  stopping: "bg-amber-100 text-amber-800", failed: "bg-red-100 text-red-700", early_stopped: "bg-emerald-100 text-emerald-800",
};

export default function Training() {
  const [configs, setConfigs] = useState<string[]>([]);
  const [jobs, setJobs] = useState<Job[]>([]);
  const [sel, setSel] = useState<string | null>(null);
  const [metrics, setMetrics] = useState<Metric[]>([]);
  const [err, setErr] = useState("");
  const [form, setForm] = useState({ name: "nano-run", config: "nano_cpu.yaml", max_steps: 300, batch_size: 8, lr: 0.001 });

  const refresh = async () => {
    try {
      const r = await get<{ jobs: Job[] }>("/training/jobs", true);
      setJobs(r.jobs); setErr("");
      const target = sel ?? r.jobs[0]?.id;
      if (target) {
        setSel(target);
        const m = await get<{ metrics: Metric[] }>(`/training/jobs/${target}/metrics`, true);
        setMetrics(m.metrics);
      }
    } catch (e) { setErr((e as Error).message); }
  };

  useEffect(() => {
    get<{ configs: string[] }>("/training/configs", true).then((r) => setConfigs(r.configs)).catch((e) => setErr(e.message));
  }, []);
  useEffect(() => {
    refresh();
    const t = setInterval(refresh, 3000);
    return () => clearInterval(t);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sel]);

  async function start() {
    try {
      const j = await post<Job>("/training/jobs", {
        name: form.name, config: form.config,
        overrides: { "training.max_steps": form.max_steps, "training.batch_size": form.batch_size, "training.lr": form.lr },
      }, true);
      setSel(j.id); refresh();
    } catch (e) { setErr((e as Error).message); }
  }
  const act = (id: string, a: "stop" | "resume") => post(`/training/jobs/${id}/${a}`, undefined, true).then(refresh).catch((e) => setErr(e.message));

  const job = jobs.find((j) => j.id === sel);
  const pr = job?.progress ?? {};
  const series = mergeSeries(metrics);
  const stat = (label: string, value: string) => <div><div className="stat-label">{label}</div><div className="stat-value">{value}</div></div>;

  return (
    <div className="space-y-4">
      {err && <div className="rounded-md bg-red-50 p-3 text-sm text-red-700">{err}{err.includes("ADMIN") || err.includes("admin") ? " — set the admin key in Settings." : ""}</div>}
      <div className="grid gap-4 lg:grid-cols-[320px_1fr]">
        <section className="card h-fit space-y-3">
          <h2 className="font-semibold">New training job</h2>
          <div className="flex flex-col gap-1"><label>Run name</label><input value={form.name} onChange={(e) => setForm({ ...form, name: e.target.value })} /></div>
          <div className="flex flex-col gap-1"><label>Config</label>
            <select value={form.config} onChange={(e) => setForm({ ...form, config: e.target.value })}>{configs.map((c) => <option key={c}>{c}</option>)}</select>
          </div>
          <div className="grid grid-cols-3 gap-2">
            <div className="flex flex-col gap-1"><label>Steps</label><input type="number" value={form.max_steps} onChange={(e) => setForm({ ...form, max_steps: Number(e.target.value) })} /></div>
            <div className="flex flex-col gap-1"><label>Batch</label><input type="number" value={form.batch_size} onChange={(e) => setForm({ ...form, batch_size: Number(e.target.value) })} /></div>
            <div className="flex flex-col gap-1"><label>LR</label><input type="number" step="0.0001" value={form.lr} onChange={(e) => setForm({ ...form, lr: Number(e.target.value) })} /></div>
          </div>
          <button className="btn-primary w-full justify-center" onClick={start}>Start training</button>
          <p className="text-xs text-stone-500">Jobs run as background processes. Stop saves a checkpoint; Resume continues from it.</p>
          <h3 className="pt-2 text-sm font-semibold">Jobs</h3>
          <ul className="space-y-1">
            {jobs.map((j) => (
              <li key={j.id}>
                <button onClick={() => setSel(j.id)} className={`flex w-full items-center justify-between rounded-md px-2 py-1.5 text-left text-sm ${sel === j.id ? "bg-stone-100" : "hover:bg-stone-50"}`}>
                  <span className="truncate">{j.name}</span>
                  <span className={`pill ${statusColor[j.status] ?? "bg-stone-100"}`}>{j.status}</span>
                </button>
              </li>
            ))}
            {jobs.length === 0 && <li className="text-sm text-stone-400">No jobs yet.</li>}
          </ul>
        </section>
        <section className="space-y-4">
          {job ? (
            <>
              <div className="card">
                <div className="mb-3 flex flex-wrap items-center gap-2">
                  <h2 className="font-semibold">{job.name}</h2>
                  <span className={`pill ${statusColor[job.status] ?? "bg-stone-100"}`}>{job.status}</span>
                  <span className="text-xs text-stone-500">{pr.model} · {pr.backend} · {pr.params ? fmt.params(pr.params) : ""} · {pr.dataset}</span>
                  <div className="ml-auto flex gap-2">
                    <button className="btn-ghost" disabled={job.status !== "running"} onClick={() => act(job.id, "stop")}>Stop</button>
                    <button className="btn-ghost" disabled={job.status === "running" || job.status === "stopping"} onClick={() => act(job.id, "resume")}>Resume</button>
                  </div>
                </div>
                <div className="grid grid-cols-2 gap-4 sm:grid-cols-4 lg:grid-cols-5">
                  {stat("Step", `${pr.step ?? 0} / ${pr.max_steps ?? "?"}`)}
                  {stat("Train loss", fmt.num(pr.train_loss, 4))}
                  {stat("Val loss", fmt.num(pr.val_loss, 4))}
                  {stat("Learning rate", pr.lr ? pr.lr.toExponential(2) : "—")}
                  {stat("Tokens", pr.tokens_seen ? pr.tokens_seen.toLocaleString() : "—")}
                  {stat("Throughput", pr.tokens_per_sec ? `${pr.tokens_per_sec.toFixed(0)} tok/s` : "—")}
                  {stat("RAM", pr.rss_mb ? `${pr.rss_mb.toFixed(0)} MB` : "—")}
                  {stat("Peak RAM", pr.peak_rss_mb ? `${pr.peak_rss_mb.toFixed(0)} MB` : "—")}
                  {stat("CPU", pr.cpu_percent !== undefined ? `${pr.cpu_percent.toFixed(0)}%` : "—")}
                  {stat("Checkpoint", pr.checkpoint_step ? `step ${pr.checkpoint_step}` : "none yet")}
                </div>
                {job.error && <pre className="mt-3 max-h-40 overflow-auto rounded bg-red-50 p-2 text-xs text-red-700">{job.error}</pre>}
              </div>
              <div className="card">
                <h3 className="mb-2 text-sm font-semibold">Loss</h3>
                <div className="h-72">
                  <ResponsiveContainer>
                    <LineChart data={series}>
                      <CartesianGrid stroke="#e7e5e4" strokeDasharray="3 3" />
                      <XAxis dataKey="step" tick={{ fontSize: 11 }} />
                      <YAxis tick={{ fontSize: 11 }} domain={["auto", "auto"]} />
                      <Tooltip />
                      <Legend />
                      <Line type="monotone" dataKey="train_loss" name="train" stroke="#1c1917" dot={false} strokeWidth={1.5} />
                      <Line type="monotone" dataKey="val_loss" name="validation" stroke="#d97706" connectNulls strokeWidth={2} />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </div>
            </>
          ) : <div className="card text-sm text-stone-500">Start a job or select one to see live progress.</div>}
        </section>
      </div>
    </div>
  );
}

function mergeSeries(ms: Metric[]) {
  const by = new Map<number, { step: number; train_loss?: number; val_loss?: number }>();
  for (const m of ms) {
    const row = by.get(m.step) ?? { step: m.step };
    if (m.type === "train") row.train_loss = m.train_loss;
    if (m.type === "eval") row.val_loss = m.val_loss;
    by.set(m.step, row);
  }
  return [...by.values()].sort((a, b) => a.step - b.step);
}
