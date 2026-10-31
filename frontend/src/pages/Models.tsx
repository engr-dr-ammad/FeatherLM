import { useEffect, useState } from "react";
import { fmt, get, ModelInfo, upload } from "../api";

export default function Models() {
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [err, setErr] = useState("");
  const [name, setName] = useState("");
  const [file, setFile] = useState<File | null>(null);
  const [msg, setMsg] = useState("");
  const [busy, setBusy] = useState(false);

  const load = () => get<{ data: { featherlm: ModelInfo }[] }>("/v1/models")
    .then((r) => { setModels(r.data.map((d) => d.featherlm)); setErr(""); })
    .catch((e) => setErr(e.message));
  useEffect(() => { load(); }, []);

  async function doImport() {
    if (!file || !name) return;
    setBusy(true); setMsg("");
    const form = new FormData();
    form.append("name", name);
    form.append("file", file);
    try {
      const r = await upload<{ id: string }>("/models/import", form);
      setMsg(`Imported as ${r.id}`);
      setName(""); setFile(null);
      load();
    } catch (e) {
      setMsg(`Import failed: ${(e as Error).message}`);
    } finally { setBusy(false); }
  }

  return (
    <div className="space-y-4">
      <section className="card overflow-x-auto">
        <div className="mb-3 flex items-center justify-between">
          <h2 className="font-semibold">Available models</h2>
          <button className="btn-ghost" onClick={load}>Refresh</button>
        </div>
        {err && <p className="text-sm text-red-600">{err}</p>}
        <table className="table">
          <thead>
            <tr>
              <th>Model</th><th>Architecture</th><th className="text-right">Parameters</th><th className="text-right">File size</th>
              <th>Quant.</th><th className="text-right">Context</th><th className="text-right">Memory est.</th><th>Training</th><th>Checkpoints</th>
            </tr>
          </thead>
          <tbody>
            {models.map((m) => (
              <tr key={m.id}>
                <td>
                  <div className="font-mono text-xs">{m.id}</div>
                  <div className="mt-0.5 flex gap-1">
                    {m.capabilities.chat && <span className="pill bg-amber-100 text-amber-800">chat</span>}
                    {m.loaded && <span className="pill bg-emerald-100 text-emerald-800">loaded</span>}
                  </div>
                </td>
                <td>{m.architecture}<div className="text-xs text-stone-500">{m.n_layers} layers · d={m.d_model}</div></td>
                <td className="text-right font-mono">{m.parameters.toLocaleString()}</td>
                <td className="text-right font-mono">{fmt.bytes(m.file_size_bytes)}</td>
                <td>{m.quantization}</td>
                <td className="text-right font-mono">{m.context_length}</td>
                <td className="text-right font-mono">~{m.memory_estimate_mb} MB</td>
                <td className="text-xs">
                  {m.training ? <>step {m.training.step}<br />val {fmt.num(m.training.best_val_loss, 3)}<br />{m.training.mode}</> : "—"}
                </td>
                <td className="text-xs text-stone-500">{m.checkpoints.join(", ")}</td>
              </tr>
            ))}
            {models.length === 0 && !err && <tr><td colSpan={9} className="py-6 text-center text-stone-400">No models yet. Train one in the Training tab or with train.py.</td></tr>}
          </tbody>
        </table>
        <p className="mt-2 text-xs text-stone-500">Memory estimate = FP32 weights + ~350 MB runtime. Quantized checkpoints are dequantized at load time by default.</p>
      </section>
      <section className="card max-w-xl space-y-3">
        <h2 className="font-semibold">Import a checkpoint</h2>
        <p className="text-sm text-stone-500">Upload a .zip containing <code>config.json</code>, <code>model.safetensors</code>, <code>tokenizer.json</code> and optionally <code>metadata.json</code>. Pickle files are rejected. Requires the admin key (Settings).</p>
        <div className="flex flex-col gap-1"><label>Name</label><input value={name} onChange={(e) => setName(e.target.value)} placeholder="my-featherlm" /></div>
        <input type="file" accept=".zip" onChange={(e) => setFile(e.target.files?.[0] ?? null)} />
        <button className="btn-primary" disabled={!file || !name || busy} onClick={doImport}>{busy ? "Importing…" : "Import"}</button>
        {msg && <p className="text-sm">{msg}</p>}
      </section>
    </div>
  );
}
