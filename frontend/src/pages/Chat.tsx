import { useEffect, useRef, useState } from "react";
import { fmt, get, ModelInfo, Perf, post, streamCompletion } from "../api";
import { storeGet, storeSet } from "./Teach";

interface Msg { role: "user" | "assistant"; content: string; perf?: Perf; finish?: string; error?: boolean }

const KEY_MODEL = "featherlm.chatModel";
const KEY_DS = "featherlm.teachDataset";

export default function Chat() {
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [model, setModel] = useState("");
  const [msgs, setMsgs] = useState<Msg[]>([]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const [p, setP] = useState({ max_tokens: 200, temperature: 0.8, top_p: 0.95, top_k: 50, repetition_penalty: 1.1, seed: "", stop: "" });
  const [remember, setRemember] = useState(storeGet("featherlm.remember") === "1");
  const [teachIdx, setTeachIdx] = useState<number | null>(null);
  const [draft, setDraft] = useState("");
  const [teachDs, setTeachDs] = useState(storeGet(KEY_DS) || "my-chatbot");
  const [teachNote, setTeachNote] = useState<{ i: number; text: string; ok: boolean } | null>(null);
  const abort = useRef<AbortController | null>(null);
  const bottom = useRef<HTMLDivElement>(null);

  useEffect(() => {
    get<{ data: { featherlm: ModelInfo }[] }>("/v1/models").then((r) => {
      const ms = r.data.map((d) => d.featherlm);
      setModels(ms);
      const wanted = storeGet(KEY_MODEL);
      const pref = ms.find((m) => m.id === wanted) ?? ms.find((m) => m.id === "runs/featherlm-nano-chat/best")
        ?? ms.find((m) => m.capabilities.chat && m.id.endsWith("/best"))
        ?? ms.find((m) => m.id.endsWith("/best")) ?? ms.find((m) => m.id.endsWith("/final")) ?? ms[0];
      if (pref) pickModel(pref.id, ms);
    }).catch(() => setModels([]));
  }, []);
  // Braces matter: newer browsers return a Promise from scrollIntoView, which React would treat as a cleanup.
  useEffect(() => { bottom.current?.scrollIntoView({ behavior: "smooth" }); }, [msgs]);

  const info = models.find((m) => m.id === model);
  const chatMode = !!info?.capabilities.chat;

  // Chat models answer best with low temperature; plain models need variety to continue text.
  function pickModel(id: string, list: ModelInfo[] = models) {
    setModel(id);
    storeSet(KEY_MODEL, id);
    const isChat = !!list.find((m) => m.id === id)?.capabilities.chat;
    setP((old) => ({ ...old, temperature: isChat ? 0 : 0.8, max_tokens: isChat ? 120 : 200, repetition_penalty: isChat ? 1.0 : 1.1 }));
    setMsgs([]);
  }

  async function saveTeach(i: number) {
    const answer = draft.trim();
    if (!answer) return;
    const prior = remember ? msgs.slice(0, i).filter((m) => !m.error) : msgs.slice(i - 1, i);
    const messages = chatMode
      ? [...prior.map(({ role, content }) => ({ role, content })), { role: "assistant", content: answer }]
      : [{ role: "user", content: msgs[i - 1]?.content ?? "" }, { role: "assistant", content: answer }];
    try {
      await post(`/teach/datasets/${encodeURIComponent(teachDs)}/examples`, { messages }, true);
      storeSet(KEY_DS, teachDs);
      setTeachNote({ i, ok: true, text: `Saved to “${teachDs}”. Train it in the Teach tab.` });
      setTeachIdx(null);
    } catch (e) {
      setTeachNote({ i, ok: false, text: `Could not save: ${(e as Error).message}` });
    }
  }

  async function send() {
    const text = input.trim();
    if (!text || busy || !model) return;
    const history: Msg[] = [...msgs, { role: "user", content: text }];
    setMsgs([...history, { role: "assistant", content: "" }]);
    setInput("");
    setBusy(true);
    const ctrl = new AbortController();
    abort.current = ctrl;
    const params = {
      model, max_tokens: p.max_tokens, temperature: p.temperature, top_p: p.top_p, top_k: p.top_k,
      repetition_penalty: p.repetition_penalty, seed: p.seed === "" ? undefined : Number(p.seed),
      stop: p.stop ? p.stop.split("|").filter(Boolean) : undefined,
    };
    // Small models answer best one question at a time; history is optional.
    const turns = remember ? history.filter((m) => !m.error) : history.slice(-1);
    const body = chatMode
      ? { ...params, messages: turns.map(({ role, content }) => ({ role, content })) }
      : { ...params, prompt: text };
    const path = chatMode ? "/v1/chat/completions" : "/v1/completions";
    const update = (fn: (m: Msg) => Msg) => setMsgs((ms) => [...ms.slice(0, -1), fn(ms[ms.length - 1])]);
    try {
      const res = await streamCompletion(path, body, (t) => update((m) => ({ ...m, content: m.content + t })), ctrl.signal);
      update((m) => ({ ...m, perf: res.perf, finish: res.finish }));
    } catch (e) {
      if ((e as Error).name === "AbortError") update((m) => ({ ...m, finish: "stopped" }));
      else update((m) => ({ ...m, content: m.content || String((e as Error).message), error: true }));
    } finally {
      setBusy(false);
      abort.current = null;
    }
  }

  const num = (k: keyof typeof p, step: number, min: number, max: number) => (
    <div className="flex flex-col gap-1">
      <label>{k.replace(/_/g, " ")}</label>
      <input type="number" step={step} min={min} max={max} value={p[k] as number}
        onChange={(e) => setP({ ...p, [k]: Number(e.target.value) })} />
    </div>
  );

  return (
    <div className="grid gap-4 lg:grid-cols-[1fr_280px]">
      <section className="card flex h-[calc(100vh-11rem)] min-h-[420px] flex-col p-0">
        <div className="flex flex-wrap items-center gap-2 border-b border-stone-200 p-3">
          <select value={model} onChange={(e) => pickModel(e.target.value)} className="min-w-0 flex-1">
            {models.length === 0 && <option value="">No models found — train one first</option>}
            {models.map((m) => <option key={m.id} value={m.id}>{m.id} · {fmt.params(m.parameters)} · {m.architecture}{m.quantization !== "none" ? ` · ${m.quantization}` : ""}</option>)}
          </select>
          <span className={`pill ${chatMode ? "bg-amber-100 text-amber-800" : "bg-stone-100 text-stone-600"}`}>
            {chatMode ? "chat template" : "completion mode"}
          </span>
          {chatMode && (
            <label className="flex items-center gap-1.5 text-xs font-normal text-stone-600" title="FeatherLM is small and answers best one question at a time">
              <input type="checkbox" checked={remember} onChange={(e) => { setRemember(e.target.checked); storeSet("featherlm.remember", e.target.checked ? "1" : "0"); }} />
              Remember earlier messages
            </label>
          )}
          <button className="btn-ghost" onClick={() => setMsgs([])} disabled={busy}>Clear</button>
        </div>
        {!chatMode && info && (
          <p className="border-b border-stone-100 bg-stone-50 px-3 py-2 text-xs text-stone-500">
            This is a plain language model without instruction tuning: it continues your text rather than following instructions.
          </p>
        )}
        <div className="flex-1 space-y-4 overflow-y-auto p-4">
          {msgs.length === 0 && (
            <p className="mt-20 text-center text-sm text-stone-400">
              {chatMode ? "Say hi, or ask what FeatherLM is, how to install it, or how to teach it. Temperature 0 gives the same, most likely answer every time." : "Enter a prompt to start generating."}
            </p>
          )}
          {msgs.map((m, i) => (
            <div key={i} className={m.role === "user" ? "flex justify-end" : ""}>
              <div className={`max-w-[85%] whitespace-pre-wrap rounded-lg px-3 py-2 text-sm ${m.role === "user" ? "bg-ink text-white" : m.error ? "bg-red-50 text-red-700" : `bg-stone-100 text-stone-800 ${chatMode ? "" : "font-mono text-[13px]"}`}`}>
                {m.role === "assistant" && !chatMode && <span className="text-stone-400">{msgs[i - 1]?.content}</span>}
                {m.content || (busy && i === msgs.length - 1 ? "…" : "")}
              </div>
              {m.perf && (
                <div className="mt-1 font-mono text-[11px] text-stone-400">
                  TTFT {m.perf.time_to_first_token_ms.toFixed(0)} ms · {m.perf.tokens_per_sec.toFixed(1)} tok/s · {(m.perf.total_time_ms / 1000).toFixed(2)} s · {m.finish}
                </div>
              )}
              {m.finish === "stopped" && !m.perf && <div className="mt-1 text-[11px] text-stone-400">stopped</div>}
              {m.role === "assistant" && !m.error && m.content && !(busy && i === msgs.length - 1) && teachIdx !== i && (
                <button className="mt-1 text-[11px] font-medium text-amber-700 hover:underline"
                  onClick={() => { setTeachIdx(i); setDraft(m.content.trim()); setTeachNote(null); }}>
                  Teach a better answer
                </button>
              )}
              {teachIdx === i && (
                <div className="mt-2 max-w-[85%] space-y-2 rounded-lg border border-amber-300 bg-amber-50 p-3">
                  <label>The answer FeatherLM should give</label>
                  <textarea rows={3} className="w-full" value={draft} onChange={(e) => setDraft(e.target.value)} autoFocus />
                  <div className="flex flex-wrap items-center gap-2">
                    <span className="text-xs text-stone-500">Save to dataset</span>
                    <input className="w-36" value={teachDs} onChange={(e) => setTeachDs(e.target.value.replace(/[^A-Za-z0-9_-]/g, "-"))} />
                    <button className="btn-primary" onClick={() => saveTeach(i)} disabled={!draft.trim()}>Save</button>
                    <button className="btn-ghost" onClick={() => setTeachIdx(null)}>Cancel</button>
                  </div>
                </div>
              )}
              {teachNote?.i === i && <div className={`mt-1 text-[11px] ${teachNote.ok ? "text-emerald-700" : "text-red-600"}`}>{teachNote.text}</div>}
            </div>
          ))}
          <div ref={bottom} />
        </div>
        <div className="flex gap-2 border-t border-stone-200 p-3">
          <textarea rows={2} className="flex-1 resize-none" placeholder={chatMode ? "Message…" : "Prompt to continue, e.g. ROMEO:"}
            value={input} onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send(); } }} />
          {busy
            ? <button className="btn-danger self-end" onClick={() => abort.current?.abort()}>Stop</button>
            : <button className="btn-primary self-end" onClick={send} disabled={!model || !input.trim()}>Send</button>}
        </div>
      </section>
      <aside className="card h-fit space-y-3 lg:sticky lg:top-20">
        <h3 className="text-sm font-semibold">Generation settings</h3>
        <div className="grid grid-cols-2 gap-3">
          {num("max_tokens", 1, 1, 4096)}
          {num("temperature", 0.05, 0, 5)}
          {num("top_p", 0.01, 0.01, 1)}
          {num("top_k", 1, 0, 10000)}
          {num("repetition_penalty", 0.05, 0.1, 5)}
          <div className="flex flex-col gap-1">
            <label>seed</label>
            <input value={p.seed} placeholder="random" onChange={(e) => setP({ ...p, seed: e.target.value.replace(/\D/g, "") })} />
          </div>
        </div>
        <div className="flex flex-col gap-1">
          <label>stop sequences (separate with |)</label>
          <input value={p.stop} placeholder="e.g. ROMEO:|THE END" onChange={(e) => setP({ ...p, stop: e.target.value })} />
        </div>
        <p className="text-xs text-stone-500">Temperature 0 gives greedy decoding. A fixed seed makes sampling reproducible.</p>
        {info && (
          <dl className="grid grid-cols-2 gap-2 border-t border-stone-100 pt-3 text-xs">
            <dt className="text-stone-500">Context</dt><dd className="font-mono">{info.context_length}</dd>
            <dt className="text-stone-500">Parameters</dt><dd className="font-mono">{info.parameters.toLocaleString()}</dd>
            <dt className="text-stone-500">Quantization</dt><dd className="font-mono">{info.quantization}</dd>
          </dl>
        )}
      </aside>
    </div>
  );
}
