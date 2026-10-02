// Unit tests for src/chatstatus.js. Playwright runs these in Node; no browser is opened.
import { expect, test } from "@playwright/test";
import { chatStatusNote, splitCode } from "../src/chatstatus.js";

const base = { configured: true, provider: "openrouter", model: "qwen/qwen3-32b", base_url_host: "openrouter.ai", ok: true, problem: null, detail: {}, checked_at: "2026-10-02T10:00:00Z" };

test("unknown status (404 or unreachable) shows nothing", () => {
  expect(chatStatusNote(null)).toEqual({ hidden: true, tone: "muted", text: "", parts: [] });
  expect(chatStatusNote(undefined).hidden).toBe(true);
});

test("not configured", () => {
  expect(chatStatusNote({ ...base, configured: false, ok: false, model: null })).toMatchObject({ hidden: false, tone: "warn", text: "Chat is not configured" });
});

test("ok shows provider and model, local for ollama", () => {
  expect(chatStatusNote(base)).toMatchObject({ tone: "muted", text: "openrouter, qwen/qwen3-32b" });
  expect(chatStatusNote({ ...base, provider: "ollama", model: "qwen3:8b" }).text).toBe("ollama, qwen3:8b (local)");
  expect(chatStatusNote({ ...base, provider: "openai", model: null }).text).toBe("openai");
});

test("each problem is shown in the warning tone", () => {
  const problems = [
    ["ollama", "Ollama is not running at localhost:11434"],
    ["openrouter", "the OpenRouter key was rejected"],
    ["openrouter", "no OpenRouter credit left"],
  ];
  for (const [provider, problem] of problems) {
    const n = chatStatusNote({ ...base, provider, ok: false, problem });
    expect(n.tone).toBe("warn");
    expect(n.text.endsWith(problem)).toBe(true);
    expect(n.parts.some((p) => "code" in p)).toBe(false);
  }
  expect(chatStatusNote({ ...base, ok: false, problem: null }).text).toBe("openrouter, qwen/qwen3-32b: The chat provider is not answering.");
});

test("a command in backticks becomes code", () => {
  const n = chatStatusNote({ ...base, provider: "ollama", model: "qwen3:8b", ok: false, problem: "model qwen3:8b is not pulled; run `ollama pull qwen3:8b`" });
  expect(n.parts).toEqual([{ text: "ollama, qwen3:8b (local): " }, { text: "model qwen3:8b is not pulled; run " }, { code: "ollama pull qwen3:8b" }]);
  expect(n.text).toBe("ollama, qwen3:8b (local): model qwen3:8b is not pulled; run ollama pull qwen3:8b");
  expect(splitCode("plain")).toEqual([{ text: "plain" }]);
});
