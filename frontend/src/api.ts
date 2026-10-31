// API client for the FeatherLM backend. Keys are kept in this browser's localStorage only.

export const keys = {
  get api() { return safeGet("featherlm.apiKey"); },
  set api(v: string) { safeSet("featherlm.apiKey", v); },
  get admin() { return safeGet("featherlm.adminKey"); },
  set admin(v: string) { safeSet("featherlm.adminKey", v); },
};

function safeGet(k: string): string {
  try { return localStorage.getItem(k) ?? ""; } catch { return ""; }
}
function safeSet(k: string, v: string) {
  try { localStorage.setItem(k, v); } catch { /* storage unavailable */ }
}

function headers(admin = false, json = true): HeadersInit {
  const h: Record<string, string> = {};
  if (json) h["Content-Type"] = "application/json";
  const k = admin ? keys.admin || keys.api : keys.api;
  if (k) h["Authorization"] = `Bearer ${k}`;
  return h;
}

export class ApiError extends Error {
  constructor(public status: number, message: string) { super(message); }
}

async function handle<T>(r: Response): Promise<T> {
  if (!r.ok) {
    let msg = r.statusText;
    try {
      const j = await r.json();
      msg = typeof j.detail === "string" ? j.detail : JSON.stringify(j.detail ?? j);
    } catch { /* not json */ }
    throw new ApiError(r.status, msg);
  }
  return r.json() as Promise<T>;
}

export const get = <T,>(path: string, admin = false) => fetch(path, { headers: headers(admin) }).then((r) => handle<T>(r));
export const post = <T,>(path: string, body?: unknown, admin = false) =>
  fetch(path, { method: "POST", headers: headers(admin), body: body === undefined ? undefined : JSON.stringify(body) }).then((r) => handle<T>(r));
export const put = <T,>(path: string, body: unknown, admin = false) =>
  fetch(path, { method: "PUT", headers: headers(admin), body: JSON.stringify(body) }).then((r) => handle<T>(r));
export const upload = <T,>(path: string, form: FormData) =>
  fetch(path, { method: "POST", headers: headers(true, false), body: form }).then((r) => handle<T>(r));

export interface ModelInfo {
  id: string; name: string; architecture: string; parameters: number; file_size_bytes: number;
  quantization: string; context_length: number; vocab_size: number; d_model: number; n_layers: number;
  chat_template: string | null; capabilities: { completions: boolean; chat: boolean };
  memory_estimate_mb: number; created?: string; checkpoints: string[]; loaded: boolean;
  training?: { step?: number; best_val_loss?: number; tokens_seen?: number; mode?: string } | null;
}

export interface Perf { time_to_first_token_ms: number; total_time_ms: number; tokens_per_sec: number; rss_mb: number }

/** Stream an OpenAI-style SSE endpoint, calling onText for each piece. Returns the final perf block. */
export async function streamCompletion(
  path: string, body: unknown, onText: (t: string) => void, signal: AbortSignal,
): Promise<{ perf?: Perf; finish?: string; usage?: { completion_tokens: number } }> {
  const r = await fetch(path, { method: "POST", headers: headers(), body: JSON.stringify({ ...(body as object), stream: true }), signal });
  if (!r.ok || !r.body) await handle(r);
  const reader = r.body!.getReader();
  const dec = new TextDecoder();
  let buf = "";
  let out: { perf?: Perf; finish?: string; usage?: { completion_tokens: number } } = {};
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buf += dec.decode(value, { stream: true });
    let idx;
    while ((idx = buf.indexOf("\n\n")) >= 0) {
      const ev = buf.slice(0, idx).trim();
      buf = buf.slice(idx + 2);
      if (!ev.startsWith("data:")) continue;
      const data = ev.slice(5).trim();
      if (data === "[DONE]") return out;
      const j = JSON.parse(data);
      if (j.error) throw new ApiError(503, j.error.message);
      const c = j.choices?.[0];
      const text = c?.delta?.content ?? c?.text ?? "";
      if (text) onText(text);
      if (j.featherlm) out = { perf: j.featherlm, finish: c?.finish_reason, usage: j.usage };
    }
  }
  return out;
}

export const fmt = {
  params: (n: number) => (n >= 1e9 ? `${(n / 1e9).toFixed(2)}B` : n >= 1e6 ? `${(n / 1e6).toFixed(1)}M` : `${(n / 1e3).toFixed(0)}K`),
  bytes: (n: number) => (n >= 2 ** 30 ? `${(n / 2 ** 30).toFixed(2)} GB` : n >= 2 ** 20 ? `${(n / 2 ** 20).toFixed(1)} MB` : `${(n / 1024).toFixed(0)} KB`),
  num: (n: number | undefined | null, d = 2) => (n === undefined || n === null || Number.isNaN(n) ? "—" : n.toFixed(d)),
};
