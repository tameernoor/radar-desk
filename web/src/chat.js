// persona chat, docked beside the viewer.
import { initAgentWidget } from "@runtypelabs/persona";
import "@runtypelabs/persona/widget.css";
import { PAGE_TOOL_NAMES } from "./pagetools.js";

const STARTERS = ["What did RADAR flag above 80%?", "Where did it score the kidney?", "Compare this run with the reference"];

// Same tokens for light and dark so the panel always matches the page.
const THEME = {
  semantic: {
    colors: {
      primary: "#e8eaed",
      accent: "#4fb3bf",
      surface: "#1b1e23",
      background: "#14161a",
      container: "#1b1e23",
      text: "#e8eaed",
      textMuted: "#9aa0a6",
      textInverse: "#14161a",
      border: "#2c3036",
      divider: "#2c3036",
    },
  },
};

// persona handles a stream-level execution_error by changing its status only, so the reason never
// reaches the panel. Read a copy of each stream and show the error's message as an assistant bubble.
async function watchForError(stream, show) {
  const reader = stream.pipeThrough(new TextDecoderStream()).getReader();
  let buffer = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += value;
    let end;
    while ((end = buffer.indexOf("\n\n")) >= 0) {
      const block = buffer.slice(0, end);
      buffer = buffer.slice(end + 2);
      const data = block.split("\n").filter((l) => l.startsWith("data:")).map((l) => l.slice(5).trim()).join("");
      if (!data.includes("execution_error")) continue;
      try {
        const frame = JSON.parse(data);
        if (frame.type === "execution_error") show(frame.error?.message || "The chat stopped with an error.");
      } catch {
        // not JSON, ignore
      }
    }
  }
}

export function mountChat({ target, context }) {
  let handle = null;
  const showError = (message) => setTimeout(() => handle?.injectAssistantMessage({ content: message }), 0);
  const customFetch = async (url, init) => {
    const res = await fetch(url, init);
    if (!res.ok || !res.body) return res;
    const [mine, theirs] = res.body.tee();
    watchForError(mine, showError).catch(() => {});
    return new Response(theirs, { status: res.status, statusText: res.statusText, headers: res.headers });
  };
  handle = initAgentWidget({
    target,
    config: {
      apiUrl: "/chat",
      customFetch,
      colorScheme: "dark",
      launcher: {
        mountMode: "docked",
        autoExpand: true,
        title: "Ask about this scan",
        subtitle: "Answers come from RADAR's scores. Research use only.",
        dock: { side: "right", width: "380px" },
      },
      welcome: {
        title: "Ask about this scan",
        subtitle: "The agent reads RADAR's 146 scores and can move the viewer. It does not diagnose.",
      },
      copy: { inputPlaceholder: "Ask about the scores or the mask" },
      suggestions: { starters: { items: STARTERS, variant: "list", behavior: "send" } },
      attachments: { enabled: false },
      voiceRecognition: { enabled: false },
      webmcp: { enabled: true, allowlist: PAGE_TOOL_NAMES, autoApprove: () => true },
      contextProviders: [() => context()],
      theme: THEME,
      darkTheme: THEME,
    },
  });
  return handle;
}
