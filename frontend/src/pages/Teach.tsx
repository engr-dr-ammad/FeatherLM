import { useEffect, useRef, useState } from "react";
import { CartesianGrid, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { get, keys, post, upload } from "../api";

interface Dataset { name: string; examples: number; builtin: boolean; updated: number }
interface Msg { role: string; content: string }
interface Example { id: string; messages: Msg[]; created?: number }
interface Job {
  id: string; name: string; status: string; error?: string | null;
  progress: { step?: number; max_steps?: number; train_loss?: number; val_loss?: number; tokens_per_sec?: number };
}
interface BaseModel { id: string; chat: boolean; parameters: number }

const KEY_DS = "featherlm.teachDataset";
export const storeGet = (k: string) => { try { return localStorage.getItem(k) ?? ""; } catch { return ""; } };
export const storeSet = (k: string, v: string) => { try { localStorage.setItem(k, v); } catch { /* unavailable */ } };

function adminHint(msg: string) {
  return /admin|403|401/i.test(msg) ? `${msg} — set the admin key in Settings, or open Studio on this computer.` : msg;
}

export default function Teach({ openChat }: { openChat: (modelId: string) => void }) {
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [ds, setDs] = useState(storeGet(KEY_DS) || "my-chatbot");
  const [examples, setExamples] = useState<Example[]>([]);
  const [q, setQ] = useState("");
  const [a, setA] = useState("");
  const [newName, setNewName] = useState("");
  const [err, setErr] = useState("");
  const [note, setNote] = useState("");
  const [bases, setBases] = useState<BaseModel[]>([]);
  const [base, setBase] = useState("");
  const [steps, setSteps] = useState(200);
  const [mode, setMode] = useState<"full" | "lora">("full");
  const [keepStarter, setKeepStarter] = useState(true);
  const [runName, setRunName] = useState("");
  const [job, setJob] = useState<Job | null>(null);
  const [modelId, setModelId] = useState("");
  const [curve, setCurve] = useState<{ step: number; loss?: number; val?: number }[]>([]);
  const fileRef = useRef<HTMLInputElement>(null);

  const current = datasets.find((d) => d.name === ds);

  async function loadDatasets() {
    try {
      const r = await get<{ datasets: Dataset[] }>("/teach/datasets", true);
      setDatasets(r.datasets);
      setErr("");
    } catch (e) { setErr(adminHint((e as Error).message)); }
  }
  async function loadExamples(name = ds) {
    if (!datasets.find((d) => d.name === name)) { setExamples([]); return; }
    try {
      const r = await get<{ examples: Example[] }>(`/teach/datasets/${encodeURIComponent(name)}/examples?limit=300`, true);
      setExamples(r.examples);
    } catch { setExamples([]); }
  }

  useEffect(() => {
    loadDatasets();
    get<{ default: string; models: BaseModel[] }>("/teach/base-models", true)
      .then((r) => { setBases(r.models); setBase(r.default ?? ""); }).catch(() => {});
  }, []);
  useEffect(() => { storeSet(KEY_DS, ds); loadExamples(ds); setRunName(suggestRun(ds)); /* eslint-disable-next-line */ }, [ds, datasets.length]);

  // Poll the running job
  useEffect(() => {
    if (!job || !["running", "pending", "stopping"].includes(job.status)) return;
    const t = setInterval(async () => {
      try {
        const j = await get<Job>(`/training/jobs/${job.id}`, true);
        setJob(j);
        const m = await get<{ metrics: { type: string; step: number; train_loss?: number; val_loss?: number }[] }>(
          `/training/jobs/${job.id}/metrics`, true);
        const by = new Map<number, { step: number; loss?: number; val?: number }>();
        for (const r of m.metrics) {
          const row = by.get(r.step) ?? { step: r.step };
          if (r.type === "train") row.loss = r.train_loss;
          if (r.type === "eval") row.val = r.val_loss;
          by.set(r.step, row);
        }
        setCurve([...by.values()].sort((x, y) => x.step - y.step));
      } catch { /* keep polling */ }
    }, 2000);
    return () => clearInterval(t);
  }, [job]);

  function suggestRun(name: string) {
    const stamp = new Date().toISOString().slice(5, 16).replace(/[-T:]/g, "");
    return `${name}-${stamp}`.slice(0, 48);
  }

  async function createDataset() {
    const name = newName.trim();
    if (!name) return;
    try {
      await post("/teach/datasets", { name }, true);
      setNewName("");
      await loadDatasets();
      setDs(name);
    } catch (e) { setErr(adminHint((e as Error).message)); }
  }

  async function addExample() {
    if (!q.trim() || !a.trim()) return;
    try {
      await post(`/teach/datasets/${encodeURIComponent(ds)}/examples`, { question: q.trim(), answer: a.trim() }, true);
      setQ(""); setA(""); setNote("Example added.");
      await loadDatasets();
      await loadExamples();
    } catch (e) { setErr(adminHint((e as Error).message)); }
  }

  async function removeExample(id: string) {
    try {
      await fetch(`/teach/datasets/${encodeURIComponent(ds)}/examples/${id}`, {
        method: "DELETE", headers: keys.admin ? { Authorization: `Bearer ${keys.admin}` } : {},
      });
      await loadDatasets();
      await loadExamples();
    } catch (e) { setErr((e as Error).message); }
  }

  async function doUpload(f: File) {
    const form = new FormData();
    form.append("file", f);
    try {
      const r = await upload<{ added: number }>(`/teach/datasets/${encodeURIComponent(ds)}/upload`, form);
      setNote(`Imported ${r.added} examples from ${f.name}.`);
      await loadDatasets();
      await loadExamples();
    } catch (e) { setErr(adminHint((e as Error).message)); }
    if (fileRef.current) fileRef.current.value = "";
  }

  async function train() {
    setErr(""); setNote(""); setCurve([]);
    try {
      const r = await post<{ job: Job; model_id: string }>("/teach/train", {
        name: runName, dataset: ds, base_model: base || undefined, steps, mode, include_starter: keepStarter,
      }, true);
      setJob(r.job);
      setModelId(r.model_id);
    } catch (e) { setErr(adminHint((e as Error).message)); }
  }

  const running = job && ["running", "pending", "stopping"].includes(job.status);
  const done = job && ["completed", "early_stopped"].includes(job.status);
  const pct = job?.progress.step && job.progress.max_steps ? Math.round((100 * job.progress.step) / job.progress.max_steps) : 0;
  const canTrain = !!current && (current.builtin || current.examples > 0) && !running && !!runName;

  return (
    <div className="space-y-4">
      <div className="rounded-xl border border-amber-200 bg-amber-50 p-4 text-sm text-amber-900">
        <p className="font-semibold">Teach FeatherLM your own answers</p>
        <ol className="mt-1 list-inside list-decimal space-y-0.5">
          <li>Add questions and the answers you want (or upload a file).</li>
          <li>Click <b>Train my chatbot</b>. It runs on this computer's CPU, usually in a few minutes.</li>
          <li>Chat with your new model. Under any reply, click <b>Teach</b> to save a better answer, then train again.</li>
        </ol>
        <p className="mt-2 text-xs text-amber-800">FeatherLM is small (13M parameters): it learns the answers you teach and close rephrasings of them. It is not a general assistant.</p>
      </div>
      {err && <div className="rounded-md bg-red-50 p-3 text-sm text-red-700">{err}</div>}
      {note && <div className="rounded-md bg-emerald-50 p-3 text-sm text-emerald-800">{note}</div>}

      <div className="grid gap-4 lg:grid-cols-[260px_1fr_340px]">
        <section className="card h-fit space-y-3">
          <h2 className="font-semibold">Datasets</h2>
          <ul className="space-y-1">
            {datasets.map((d) => (
              <li key={d.name}>
                <button onClick={() => setDs(d.name)}
                  className={`flex w-full items-center justify-between rounded-md px-2 py-1.5 text-left text-sm ${ds === d.name ? "bg-stone-100 font-medium" : "hover:bg-stone-50"}`}>
                  <span className="truncate">{d.name}{d.builtin && <span className="ml-1 text-xs text-stone-400">built-in</span>}</span>
                  <span className="text-xs text-stone-500">{d.examples}</span>
                </button>
              </li>
            ))}
            {!datasets.some((d) => d.name === ds) && (
              <li className="px-2 py-1.5 text-sm text-stone-500">{ds} <span className="text-xs">(created when you add an example)</span></li>
            )}
          </ul>
          <div className="flex gap-2">
            <input className="min-w-0 flex-1" placeholder="new dataset name" value={newName} onChange={(e) => setNewName(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && createDataset()} />
            <button className="btn-ghost" onClick={createDataset}>Add</button>
          </div>
        </section>

        <section className="space-y-4">
          <div className="card space-y-3">
            <h2 className="font-semibold">Add an example to <span className="text-amber-700">{ds}</span></h2>
            {current?.builtin ? (
              <p className="text-sm text-stone-500">The built-in starter conversations are read-only. Create your own dataset to add examples.</p>
            ) : (
              <>
                <div className="flex flex-col gap-1"><label>When someone asks…</label>
                  <input value={q} onChange={(e) => setQ(e.target.value)} placeholder="e.g. What are your opening hours?" /></div>
                <div className="flex flex-col gap-1"><label>FeatherLM should answer…</label>
                  <textarea rows={3} value={a} onChange={(e) => setA(e.target.value)} placeholder="e.g. We're open Monday to Friday, 9am to 5pm." /></div>
                <div className="flex flex-wrap items-center gap-2">
                  <button className="btn-primary" onClick={addExample} disabled={!q.trim() || !a.trim()}>Add example</button>
                  <span className="text-xs text-stone-400">or</span>
                  <input ref={fileRef} type="file" accept=".jsonl,.json,.csv,.txt" className="max-w-[230px] text-xs"
                    onChange={(e) => e.target.files?.[0] && doUpload(e.target.files[0])} />
                </div>
                <p className="text-xs text-stone-500">Files: <code>.csv</code> with question,answer columns · <code>.txt</code> with <code>Q:</code>/<code>A:</code> blocks · <code>.jsonl</code> chat messages.</p>
              </>
            )}
          </div>
          <div className="card">
            <h3 className="mb-2 text-sm font-semibold">Examples ({current?.examples ?? 0}){current?.builtin && current.examples > 300 && <span className="font-normal text-stone-400"> · showing 300</span>}</h3>
            <ul className="max-h-[420px] divide-y divide-stone-100 overflow-y-auto">
              {examples.map((ex) => {
                const user = [...ex.messages].reverse().find((m) => m.role === "user");
                const ans = ex.messages[ex.messages.length - 1];
                return (
                  <li key={ex.id} className="flex gap-3 py-2 text-sm">
                    <div className="min-w-0 flex-1">
                      <p className="font-medium text-stone-800">{user?.content}</p>
                      <p className="text-stone-600">→ {ans?.content}</p>
                    </div>
                    {!current?.builtin && <button className="text-xs text-stone-400 hover:text-red-600" onClick={() => removeExample(ex.id)}>Delete</button>}
                  </li>
                );
              })}
              {examples.length === 0 && <li className="py-6 text-center text-sm text-stone-400">No examples yet.</li>}
            </ul>
          </div>
        </section>

        <section className="card h-fit space-y-3">
          <h2 className="font-semibold">Train my chatbot</h2>
          <div className="flex flex-col gap-1"><label>Name for the new model</label>
            <input value={runName} onChange={(e) => setRunName(e.target.value.replace(/[^A-Za-z0-9_-]/g, "-"))} /></div>
          <div className="flex flex-col gap-1"><label>Start from</label>
            <select value={base} onChange={(e) => setBase(e.target.value)}>
              {bases.map((b) => <option key={b.id} value={b.id}>{b.id}{b.chat ? " · chat" : ""}</option>)}
            </select></div>
          <div className="flex flex-col gap-1">
            <label>Training steps: {steps} <span className="text-stone-400">(~{Math.max(1, Math.round(steps * 0.7 / 60))} min on 2 cores)</span></label>
            <input type="range" min={50} max={800} step={50} value={steps} onChange={(e) => setSteps(Number(e.target.value))} className="border-0 px-0" />
          </div>
          <div className="flex gap-2 text-sm">
            {(["full", "lora"] as const).map((m) => (
              <button key={m} onClick={() => setMode(m)} className={`flex-1 rounded-md border px-2 py-1.5 ${mode === m ? "border-ink bg-ink text-white" : "border-stone-300 bg-white"}`}>
                {m === "full" ? "Full (best)" : "LoRA (lighter)"}
              </button>
            ))}
          </div>
          <label className="flex items-start gap-2 text-sm font-normal text-stone-700">
            <input type="checkbox" checked={keepStarter} onChange={(e) => setKeepStarter(e.target.checked)} className="mt-1" />
            Keep what it already knows (mix in the starter conversations)
          </label>
          <button className="btn-primary w-full justify-center" disabled={!canTrain} onClick={train}>
            {running ? "Training…" : "Train my chatbot"}
          </button>
          {job && (
            <div className="space-y-2 border-t border-stone-100 pt-3">
              <div className="flex items-center justify-between text-sm">
                <span className="font-medium">{job.name}</span>
                <span className={`pill ${done ? "bg-emerald-100 text-emerald-800" : job.status === "failed" ? "bg-red-100 text-red-700" : "bg-blue-100 text-blue-800"}`}>{job.status}</span>
              </div>
              <div className="h-2 overflow-hidden rounded-full bg-stone-200"><div className="h-full bg-amber-500 transition-all" style={{ width: `${done ? 100 : pct}%` }} /></div>
              <p className="font-mono text-xs text-stone-500">step {job.progress.step ?? 0}/{job.progress.max_steps ?? steps} · loss {job.progress.train_loss?.toFixed(3) ?? "—"}</p>
              {curve.length > 1 && (
                <div className="h-28">
                  <ResponsiveContainer>
                    <LineChart data={curve}>
                      <CartesianGrid stroke="#e7e5e4" strokeDasharray="3 3" />
                      <XAxis dataKey="step" tick={{ fontSize: 10 }} />
                      <YAxis tick={{ fontSize: 10 }} width={28} />
                      <Tooltip />
                      <Line dataKey="loss" stroke="#1c1917" dot={false} strokeWidth={1.5} />
                      <Line dataKey="val" stroke="#d97706" connectNulls strokeWidth={2} />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              )}
              {job.error && <pre className="max-h-32 overflow-auto rounded bg-red-50 p-2 text-xs text-red-700">{job.error}</pre>}
              {done && <button className="btn-primary w-full justify-center" onClick={() => openChat(modelId)}>Chat with {job.name}</button>}
            </div>
          )}
        </section>
      </div>
    </div>
  );
}
