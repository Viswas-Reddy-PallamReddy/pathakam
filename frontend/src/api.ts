export type Lang = "auto" | "te" | "en";

export interface Source {
  n: number;
  doc_id: string;
  title: string;
  page: number;
  url: string | null;
  snippet: string;
}

export interface Done {
  grounded: boolean;
  retrieval_ms: number;
  first_token_ms?: number;
  total_ms: number;
}

export type ChatEvent =
  | { type: "meta"; lang: "te" | "en"; model: string }
  | { type: "sources"; sources: Source[] }
  | { type: "token"; text: string }
  | ({ type: "done" } & Done)
  | { type: "error"; message: string };

export interface Health {
  status: string;
  llm: string | null;
  llm_error: string | null;
  embedder: string | null;
  documents: number;
  chunks: number;
}

export async function getHealth(): Promise<Health> {
  const r = await fetch("/api/health");
  if (!r.ok) throw new Error(`health ${r.status}`);
  return r.json();
}

/** POST /api/chat and parse the server-sent event stream. */
export async function streamChat(
  body: { question: string; history: { role: string; content: string }[]; lang: Lang },
  onEvent: (e: ChatEvent) => void,
  signal: AbortSignal,
): Promise<void> {
  const r = await fetch("/api/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
    signal,
  });
  if (!r.ok || !r.body) throw new Error(`Server returned ${r.status}`);
  const reader = r.body.pipeThrough(new TextDecoderStream()).getReader();
  let buf = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buf += value;
    let sep;
    while ((sep = buf.indexOf("\n\n")) >= 0) {
      const block = buf.slice(0, sep);
      buf = buf.slice(sep + 2);
      const data = block
        .split("\n")
        .filter((l) => l.startsWith("data:"))
        .map((l) => l.slice(5).trim())
        .join("");
      if (data) onEvent(JSON.parse(data) as ChatEvent);
    }
  }
}
