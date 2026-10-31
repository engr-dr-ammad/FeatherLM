import { Fragment, useEffect, useState } from "react";
import { get, keys, put } from "../api";

interface S {
  threads: number; default_model: string; max_new_tokens: number; training_memory_budget_mb: number; log_level: string;
  models_dir: string; runs_dir: string; training_api: boolean; auth_enabled: boolean; admin_key_set: boolean;
  max_loaded_models: number; max_training_jobs: number;
}

export default function SettingsPage() {
  const [s, setS] = useState<S | null>(null);
  const [apiKey, setApiKey] = useState(keys.api);
  const [adminKey, setAdminKey] = useState(keys.admin);
  const [sys, setSys] = useState<Record<string, number | string | boolean> | null>(null);
  const [msg, setMsg] = useState("");

  const load = () => {
    get<S>("/settings").then(setS).catch((e) => setMsg(e.message));
    get<Record<string, number | string | boolean>>("/system").then(setSys).catch(() => {});
  };
  useEffect(load, []);

  async function save() {
    if (!s) return;
    try {
      const r = await put<S>("/settings", {
        threads: s.threads, default_model: s.default_model, max_new_tokens: s.max_new_tokens,
        training_memory_budget_mb: s.training_memory_budget_mb, log_level: s.log_level,
      }, true);
      setS(r); setMsg("Saved.");
    } catch (e) { setMsg(`Save failed: ${(e as Error).message}`); }
  }
  function saveKeys() { keys.api = apiKey; keys.admin = adminKey; setMsg("Keys stored in this browser."); load(); }

  const field = (label: string, el: React.ReactNode) => <div className="flex flex-col gap-1"><label>{label}</label>{el}</div>;

  return (
    <div className="grid gap-4 lg:grid-cols-2">
      <section className="card space-y-3">
        <h2 className="font-semibold">Server settings</h2>
        {s ? (
          <>
            <div className="grid grid-cols-2 gap-3">
              {field("CPU threads (0 = all)", <input type="number" min={0} value={s.threads} onChange={(e) => setS({ ...s, threads: Number(e.target.value) })} />)}
              {field("Max new tokens", <input type="number" min={1} value={s.max_new_tokens} onChange={(e) => setS({ ...s, max_new_tokens: Number(e.target.value) })} />)}
              {field("Training memory budget (MB)", <input type="number" min={256} value={s.training_memory_budget_mb} onChange={(e) => setS({ ...s, training_memory_budget_mb: Number(e.target.value) })} />)}
              {field("Log level", <select value={s.log_level} onChange={(e) => setS({ ...s, log_level: e.target.value })}>{["DEBUG", "INFO", "WARNING", "ERROR"].map((l) => <option key={l}>{l}</option>)}</select>)}
            </div>
            {field("Default model id", <input value={s.default_model} placeholder="e.g. runs/featherlm-nano/final" onChange={(e) => setS({ ...s, default_model: e.target.value })} />)}
            <dl className="grid grid-cols-2 gap-2 text-xs">
              <dt className="text-stone-500">Model storage</dt><dd className="font-mono">{s.models_dir}</dd>
              <dt className="text-stone-500">Training output</dt><dd className="font-mono">{s.runs_dir}</dd>
              <dt className="text-stone-500">Training API</dt><dd>{s.training_api ? "enabled" : "disabled"}</dd>
              <dt className="text-stone-500">API auth</dt><dd>{s.auth_enabled ? "required" : "off (local mode)"}</dd>
              <dt className="text-stone-500">Admin key</dt><dd>{s.admin_key_set ? "configured" : "not set (admin only from localhost)"}</dd>
            </dl>
            <button className="btn-primary" onClick={save}>Save settings</button>
            <p className="text-xs text-stone-500">Paths, keys and limits are set with environment variables on the server (see .env.example).</p>
          </>
        ) : <p className="text-sm text-stone-500">Loading…</p>}
      </section>
      <section className="space-y-4">
        <div className="card space-y-3">
          <h2 className="font-semibold">API keys (this browser)</h2>
          {field("API key", <input type="password" value={apiKey} onChange={(e) => setApiKey(e.target.value)} placeholder="FEATHERLM_API_KEY" />)}
          {field("Admin key", <input type="password" value={adminKey} onChange={(e) => setAdminKey(e.target.value)} placeholder="FEATHERLM_ADMIN_KEY" />)}
          <button className="btn-ghost" onClick={saveKeys}>Store keys</button>
        </div>
        {sys && (
          <div className="card">
            <h2 className="mb-2 font-semibold">System</h2>
            <dl className="grid grid-cols-2 gap-2 text-xs">
              {Object.entries(sys).map(([k, v]) => <Fragment key={k}><dt className="text-stone-500">{k.replace(/_/g, " ")}</dt><dd className="font-mono">{String(v)}</dd></Fragment>)}
            </dl>
          </div>
        )}
        {msg && <p className="text-sm">{msg}</p>}
      </section>
    </div>
  );
}
