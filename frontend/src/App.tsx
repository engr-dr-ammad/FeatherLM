import { useEffect, useState } from "react";
import { get } from "./api";
import Chat from "./pages/Chat";
import Teach, { storeSet } from "./pages/Teach";
import Models from "./pages/Models";
import Training from "./pages/Training";
import Benchmarks from "./pages/Benchmarks";
import SettingsPage from "./pages/Settings";

const TABS = [
  { id: "chat", label: "Chat" },
  { id: "teach", label: "Teach" },
  { id: "models", label: "Model Manager" },
  { id: "training", label: "Training" },
  { id: "benchmarks", label: "Benchmarks" },
  { id: "settings", label: "Settings" },
] as const;
type Tab = (typeof TABS)[number]["id"];

function initialTab(): Tab {
  const h = window.location.hash.replace("#", "");
  return (TABS.find((t) => t.id === h)?.id ?? "chat") as Tab;
}

export default function App() {
  const [tab, setTab] = useState<Tab>(initialTab);
  const [health, setHealth] = useState<{ status: string; version: string } | null>(null);

  useEffect(() => {
    const ping = () => get<{ status: string; version: string }>("/health").then(setHealth).catch(() => setHealth(null));
    ping();
    const t = setInterval(ping, 10000);
    return () => clearInterval(t);
  }, []);

  const go = (t: Tab) => { setTab(t); window.location.hash = t; };

  return (
    <div className="flex min-h-full flex-col">
      <header className="sticky top-0 z-10 border-b border-stone-200 bg-paper/90 backdrop-blur">
        <div className="mx-auto flex max-w-7xl flex-wrap items-center gap-x-6 gap-y-2 px-4 py-3">
          <div className="flex items-center gap-2">
            <img src="/favicon.svg" alt="" className="h-7 w-7" />
            <span className="text-lg font-semibold tracking-tight text-ink">FeatherLM <span className="font-normal text-stone-500">Studio</span></span>
          </div>
          <nav className="-mx-1 flex flex-1 gap-1 overflow-x-auto">
            {TABS.map((t) => (
              <button key={t.id} onClick={() => go(t.id)}
                className={`whitespace-nowrap rounded-md px-3 py-1.5 text-sm transition ${tab === t.id ? "bg-ink text-white" : "text-stone-600 hover:bg-stone-200/60"}`}>
                {t.label}
              </button>
            ))}
          </nav>
          <span className={`pill ${health ? "bg-emerald-100 text-emerald-800" : "bg-red-100 text-red-700"}`}>
            {health ? `API ${health.version}` : "API offline"}
          </span>
        </div>
      </header>
      <main className="mx-auto w-full max-w-7xl flex-1 px-4 py-5">
        {tab === "chat" && <Chat />}
        {tab === "teach" && <Teach openChat={(id) => { storeSet("featherlm.chatModel", id); go("chat"); }} />}
        {tab === "models" && <Models />}
        {tab === "training" && <Training />}
        {tab === "benchmarks" && <Benchmarks />}
        {tab === "settings" && <SettingsPage />}
      </main>
      <footer className="border-t border-stone-200 py-4 text-center text-xs text-stone-500">
        FeatherLM — Developed by Ammad Aslam with AI-assisted tooling.
      </footer>
    </div>
  );
}
