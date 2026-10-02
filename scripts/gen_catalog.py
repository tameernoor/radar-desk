"""Generate src/radar_desk/radar/catalog.json from the vendored DAMO inference_demo.py.

Reads the source with ast, so torch is never imported. The English label names
come from the commented-out organ_dict in the same file.

Usage: python scripts/gen_catalog.py
"""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "worker" / "vendor" / "damo-radar" / "RADAR_inference" / "inference_demo.py"
OUTPUT = ROOT / "src" / "radar_desk" / "radar" / "catalog.json"
VENDORED = ROOT / "worker" / "vendor" / "damo-radar" / "VENDORED.md"


def source_commit() -> str:
    """The pinned upstream commit, read from VENDORED.md so a re-vendor cannot leave it stale."""
    match = re.search(r"Commit `([0-9a-f]{40})`", VENDORED.read_text())
    if not match:
        raise SystemExit(f"no commit line in {VENDORED}")
    return match.group(1)


def _datafolder_attrs(tree: ast.Module) -> dict:
    """The literal `self.<name> = ...` assignments in DataFolder.__init__."""
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "DataFolder")
    init = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__init__")
    attrs = {}
    for node in ast.walk(init):
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            t = node.targets[0]
            if isinstance(t, ast.Attribute) and isinstance(t.value, ast.Name) and t.value.id == "self":
                try:
                    attrs[t.attr] = ast.literal_eval(node.value)
                except ValueError:
                    pass
    return attrs


def _commented_organ_dict(text: str) -> dict[str, str]:
    """The first `# organ_dict = {...}` block, uncommented and evaluated."""
    lines = text.splitlines()
    start = next(i for i, line in enumerate(lines) if line.strip().startswith("# organ_dict = {"))
    body = []
    for line in lines[start:]:
        stripped = line.strip()
        if not stripped.startswith("#"):
            break
        body.append(stripped[1:])
        if stripped[1:].strip() == "}":
            break
    src = "\n".join(body).split("=", 1)[1]
    return ast.literal_eval(src)


def build_catalog(source: Path = SOURCE) -> dict:
    text = source.read_text(encoding="utf-8")
    attrs = _datafolder_attrs(ast.parse(text))
    organs: list[str] = attrs["organs"]
    test_items: list[str] = attrs["test_items"]
    english: dict[str, str] = attrs["english_mapping"]
    organ_en = _commented_organ_dict(text)

    if len(organs) != 36 or len(test_items) != 146 or set(english) != set(test_items):
        raise ValueError("vendored DataFolder lists have an unexpected shape")

    findings = []
    organ_names: dict[str, str] = {}
    for i, key in enumerate(test_items):
        organ_zh, _ = key.split("_", 1)
        organ, finding = english[key].split("_", 1)
        if organ_names.setdefault(organ_zh, organ) != organ:
            raise ValueError(f"organ {organ_zh} has two English names")
        findings.append(
            {"index": i, "key": key, "organ_zh": organ_zh, "organ": organ, "finding": finding,
             "english": english[key]}
        )

    labels = [{"label": 0, "zh": "background", "en": "background"}]
    labels += [{"label": i + 1, "zh": zh, "en": organ_en[zh]} for i, zh in enumerate(organs)]

    scored = []
    for organ_zh, organ in organ_names.items():  # insertion order = first appearance
        scored.append(
            {"organ": organ, "organ_zh": organ_zh, "label": organs.index(organ_zh) + 1,
             "finding_count": sum(1 for f in findings if f["organ_zh"] == organ_zh)}
        )

    return {"source_commit": source_commit(), "findings": findings, "labels": labels, "scored_organs": scored}


def render(catalog: dict) -> str:
    return json.dumps(catalog, sort_keys=True, indent=2, ensure_ascii=False) + "\n"


def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(render(build_catalog()), encoding="utf-8")
    print(f"wrote {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
