// The one-line provider note in the chat panel, from GET /chat/status. Pure: no DOM.
// Returns {hidden, tone: "muted" | "warn", text, parts}, where parts are {text} or {code}
// pieces so a command in backticks can be shown as code.

// "run `ollama pull qwen3:8b`" -> [{text: "run "}, {code: "ollama pull qwen3:8b"}]
export function splitCode(message) {
  return String(message)
    .split(/`([^`]+)`/)
    .map((piece, i) => (i % 2 ? { code: piece } : { text: piece }))
    .filter((p) => (p.code ?? p.text) !== "");
}

function providerLabel(status) {
  const name = status.provider || "chat";
  if (!status.model) return name;
  return `${name}, ${status.model}${status.provider === "ollama" ? " (local)" : ""}`;
}

function note(tone, parts) {
  const text = parts.map((p) => p.code ?? p.text).join("");
  return { hidden: false, tone, text, parts };
}

// status is the parsed response, or null when the route is missing (404) or the request failed.
export function chatStatusNote(status) {
  if (!status || typeof status !== "object") return { hidden: true, tone: "muted", text: "", parts: [] };
  if (status.configured === false) return note("warn", [{ text: "Chat is not configured" }]);
  if (status.ok === false) {
    const problem = status.problem || "The chat provider is not answering.";
    return note("warn", [{ text: `${providerLabel(status)}: ` }, ...splitCode(problem)]);
  }
  return note("muted", [{ text: providerLabel(status) }]);
}
