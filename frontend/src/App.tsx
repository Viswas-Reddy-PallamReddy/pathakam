import { Fragment, useEffect, useRef, useState } from "react";
import { getHealth, streamChat, type Done, type Health, type Lang, type Source } from "./api";

interface Turn {
  id: number;
  role: "user" | "assistant";
  text: string;
  sources?: Source[];
  done?: Done;
  error?: string;
  pending?: boolean;
}

const EXAMPLES: Record<"te" | "en", string[]> = {
  te: ["ఈ పథకానికి ఎవరు అర్హులు?", "దరఖాస్తు చేయడానికి ఏ పత్రాలు కావాలి?", "దరఖాస్తు ఎక్కడ చేయాలి?"],
  en: ["Who is eligible for this scheme?", "Which documents do I need to apply?", "Where do I apply?"],
};

const UI = {
  te: { placeholder: "మీ ప్రశ్నను తెలుగులో లేదా ఇంగ్లీషులో అడగండి…", send: "అడగండి", stop: "ఆపండి", sources: "ఆధారాలు", page: "పేజీ" },
  en: { placeholder: "Ask in Telugu or English…", send: "Ask", stop: "Stop", sources: "Sources", page: "page" },
};

function sourceHref(s: Source): string | undefined {
  if (!s.url) return undefined;
  return /\.pdf($|\?)/i.test(s.url) ? `${s.url}#page=${s.page}` : s.url;
}

/** Render answer text, turning [n] into buttons that highlight source n. */
function AnswerText({ text, onCite }: { text: string; onCite: (n: number) => void }) {
  const parts = text.split(/(\[\d+\])/g);
  return (
    <p className="answer">
      {parts.map((p, i) => {
        const m = /^\[(\d+)\]$/.exec(p);
        return m ? (
          <button key={i} className="cite" onClick={() => onCite(Number(m[1]))} aria-label={`source ${m[1]}`}>
            {m[1]}
          </button>
        ) : (
          <Fragment key={i}>{p}</Fragment>
        );
      })}
    </p>
  );
}

function fmtMs(ms?: number) {
  if (ms === undefined) return "–";
  return ms < 1000 ? `${Math.round(ms)} ms` : `${(ms / 1000).toFixed(1)} s`;
}

export default function App() {
  const [turns, setTurns] = useState<Turn[]>([]);
  const [input, setInput] = useState("");
  const [lang, setLang] = useState<Lang>("auto");
  const [health, setHealth] = useState<Health | null>(null);
  const [healthErr, setHealthErr] = useState(false);
  const [highlight, setHighlight] = useState<{ turn: number; n: number } | null>(null);
  const abortRef = useRef<AbortController | null>(null);
  const endRef = useRef<HTMLDivElement>(null);
  const busy = turns.some((t) => t.pending);
  const uiLang = lang === "en" ? "en" : "te";
  const ui = UI[uiLang];

  useEffect(() => {
    getHealth().then(setHealth).catch(() => setHealthErr(true));
  }, []);
  useEffect(() => endRef.current?.scrollIntoView({ behavior: "smooth" }), [turns]);

  const patch = (id: number, f: (t: Turn) => Turn) => setTurns((ts) => ts.map((t) => (t.id === id ? f(t) : t)));

  async function ask(question: string) {
    question = question.trim();
    if (!question || busy) return;
    const history = turns
      .filter((t) => !t.error && !t.pending && t.text)
      .map((t) => ({ role: t.role, content: t.text }));
    const userId = Date.now();
    const botId = userId + 1;
    setTurns((ts) => [...ts, { id: userId, role: "user", text: question }, { id: botId, role: "assistant", text: "", pending: true }]);
    setInput("");
    const ctrl = new AbortController();
    abortRef.current = ctrl;
    try {
      await streamChat(
        { question, history, lang },
        (e) => {
          if (e.type === "sources") patch(botId, (t) => ({ ...t, sources: e.sources }));
          else if (e.type === "token") patch(botId, (t) => ({ ...t, text: t.text + e.text }));
          else if (e.type === "done") patch(botId, (t) => ({ ...t, done: e, pending: false }));
          else if (e.type === "error") patch(botId, (t) => ({ ...t, error: e.message, pending: false }));
        },
        ctrl.signal,
      );
    } catch (err) {
      if (!ctrl.signal.aborted) patch(botId, (t) => ({ ...t, error: String(err) }));
    } finally {
      patch(botId, (t) => ({ ...t, pending: false }));
    }
  }

  return (
    <div className="app">
      <header>
        <div className="brand">
          <span className="logo" aria-hidden>ప</span>
          <div>
            <h1>Pathakam</h1>
            <p className="tagline">పథకం సహాయకుడు · Welfare scheme helpdesk</p>
          </div>
        </div>
        <div className="controls">
          <select value={lang} onChange={(e) => setLang(e.target.value as Lang)} aria-label="Answer language">
            <option value="auto">Auto</option>
            <option value="te">తెలుగు</option>
            <option value="en">English</option>
          </select>
          <span className={`status ${health?.status === "ok" ? "ok" : "bad"}`} title={health?.llm_error ?? ""}>
            {healthErr ? "backend offline" : health ? `${health.llm ?? "no model"} · ${health.documents} docs` : "…"}
          </span>
        </div>
      </header>

      <main>
        {turns.length === 0 && (
          <section className="empty">
            <p>{uiLang === "te" ? "అధికారిక పత్రాల ఆధారంగా సమాధానాలు, ప్రతి వాక్యానికి ఆధారంతో." : "Answers from official documents, with a source for every fact."}</p>
            <div className="examples">
              {[...EXAMPLES.te, ...EXAMPLES.en].map((q) => (
                <button key={q} onClick={() => ask(q)}>{q}</button>
              ))}
            </div>
          </section>
        )}

        {turns.map((t) => (
          <article key={t.id} className={`turn ${t.role}`}>
            {t.role === "user" ? (
              <p>{t.text}</p>
            ) : (
              <>
                {t.text ? (
                  <AnswerText text={t.text} onCite={(n) => setHighlight({ turn: t.id, n })} />
                ) : (
                  t.pending && <p className="thinking">…</p>
                )}
                {t.error && <p className="error">{t.error}</p>}
                {t.sources && t.sources.length > 0 && (
                  <details className="sources" open={highlight?.turn === t.id}>
                    <summary>{ui.sources} ({t.sources.length})</summary>
                    <ol>
                      {t.sources.map((s) => (
                        <li key={s.n} className={highlight?.turn === t.id && highlight.n === s.n ? "hl" : ""}>
                          <span className="n">{s.n}</span>
                          <div>
                            <a href={sourceHref(s)} target="_blank" rel="noreferrer">
                              {s.title}
                            </a>{" "}
                            <span className="page">· {ui.page} {s.page}</span>
                            <p className="snippet">{s.snippet}</p>
                          </div>
                        </li>
                      ))}
                    </ol>
                  </details>
                )}
                {t.done && (
                  <p className="timing">
                    retrieval {fmtMs(t.done.retrieval_ms)} · first token {fmtMs(t.done.first_token_ms)} · total {fmtMs(t.done.total_ms)}
                  </p>
                )}
              </>
            )}
          </article>
        ))}
        <div ref={endRef} />
      </main>

      <form
        className="composer"
        onSubmit={(e) => {
          e.preventDefault();
          ask(input);
        }}
      >
        <textarea
          value={input}
          rows={2}
          maxLength={2000}
          placeholder={ui.placeholder}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey) {
              e.preventDefault();
              ask(input);
            }
          }}
        />
        {busy ? (
          <button type="button" onClick={() => abortRef.current?.abort()}>{ui.stop}</button>
        ) : (
          <button type="submit" disabled={!input.trim()}>{ui.send}</button>
        )}
      </form>
      <footer>
        Not an official government service. Confirm details at your village/ward secretariat or MeeSeva centre before applying.
      </footer>
    </div>
  );
}
