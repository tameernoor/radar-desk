"""Static checks on the GPU worker: it cannot run here (no CUDA, no Modal account), so check what can be checked."""

from __future__ import annotations

import ast
import importlib
import json
import socket
import sys
import warnings
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
WORKER = ROOT / "worker"
SCRIPTS = [ROOT / "scripts" / n for n in ("weights_check.py", "modal_spike.py", "parity.py", "score_local.py")]
PY310_FILES = sorted((WORKER / "radar_worker").glob("*.py")) + [WORKER / "modal_app.py"] + SCRIPTS

EXPECTED_WEIGHTS = {
    "checkpoint_radar_pretrain.pth": 1566049482,
    "infer_text_embedding_radar.pt": 346348,
    "bert-base-chinese/config.json": 656,
    "bert-base-chinese/config_decoder.json": 711,
    "bert-base-chinese/pytorch_model.bin": 411577189,
    "bert-base-chinese/tokenizer.json": 268943,
    "bert-base-chinese/tokenizer_config.json": 49,
    "bert-base-chinese/vocab.txt": 109540,
}


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            names.add(node.module)
    return names


@pytest.mark.parametrize("path", PY310_FILES, ids=lambda p: str(p.relative_to(ROOT)))
def test_parses_as_python_310(path):
    assert path.is_file(), path
    ast.parse(path.read_text(encoding="utf-8"), filename=str(path), feature_version=(3, 10))


def test_weights_manifest_lists_the_eight_files():
    data = json.loads((WORKER / "weights.json").read_text())
    assert data["volume"] == "radar-weights"
    got = {f["path"]: f["size"] for f in data["files"]}
    assert got == EXPECTED_WEIGHTS
    for f in data["files"]:
        assert len(f["sha256"]) == 64 and int(f["sha256"], 16) >= 0
    assert "sources" in data


def test_vendored_text_embedding_matches_manifest():
    import hashlib

    data = json.loads((WORKER / "weights.json").read_text())
    entry = next(f for f in data["files"] if f["path"] == "infer_text_embedding_radar.pt")
    blob = (WORKER / "vendor/damo-radar/ckpt/infer_text_embedding_radar.pt").read_bytes()
    assert len(blob) == entry["size"]
    assert hashlib.sha256(blob).hexdigest() == entry["sha256"]


def test_worker_never_imports_the_api_package():
    for path in sorted((WORKER / "radar_worker").glob("*.py")) + [WORKER / "modal_app.py"]:
        assert not any(n == "radar_desk" or n.startswith("radar_desk.") for n in _imports(path)), path


def test_geometry_and_io_need_only_numpy_nibabel_stdlib():
    allowed_third_party = {"numpy", "nibabel", "nibabel.orientations"}
    for name in ("geometry.py", "io.py", "weights.py", "__init__.py"):
        for mod in _imports(WORKER / "radar_worker" / name):
            top = mod.split(".")[0]
            assert mod in allowed_third_party or top in sys.stdlib_module_names, (name, mod)


def test_infer_imports_torch_and_monai_only_inside_functions():
    tree = ast.parse((WORKER / "radar_worker" / "infer.py").read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            mods = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module or ""]
            for m in mods:
                assert m.split(".")[0] not in {"torch", "monai", "inference_demo", "transformers"}, m


@pytest.mark.parametrize("name", ["pull.py", "job.py"])
def test_pull_and_job_import_heavy_modules_only_inside_functions(name):
    path = WORKER / "radar_worker" / name
    assert not any(n == "radar_desk" or n.startswith("radar_desk.") for n in _imports(path)), name
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Import):
            mods = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            mods = [node.module or ""] + [f"{node.module}.{a.name}" for a in node.names]
        else:
            continue
        for m in mods:
            assert m.split(".")[0] not in {"torch", "monai"} and not m.startswith("radar_worker.infer"), (name, m)


def _top_level_imports(path: Path) -> list[str]:
    """Module names imported at module level (not inside functions), with `from x import y` as x and x.y."""
    mods = []
    for node in ast.parse(path.read_text(encoding="utf-8")).body:
        if isinstance(node, ast.Import):
            mods += [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            mods += [node.module or ""] + [f"{node.module}.{a.name}" for a in node.names]
    return mods


def test_serverless_and_volume_are_parsed_as_python_310():
    names = {p.name for p in PY310_FILES}
    assert {"serverless.py", "volume.py"} <= names


@pytest.mark.parametrize("name", ["serverless.py", "volume.py"])
def test_serverless_and_volume_never_import_the_api_package(name):
    assert not any(n == "radar_desk" or n.startswith("radar_desk.") for n in _imports(WORKER / "radar_worker" / name))


def test_serverless_imports_runpod_and_torch_only_inside_functions():
    path = WORKER / "radar_worker" / "serverless.py"
    for m in _top_level_imports(path):
        assert m.split(".")[0] not in {"runpod", "torch", "monai"} and not m.startswith("radar_worker.infer"), m
    assert "runpod" in _imports(path)  # main() imports it


def test_volume_imports_only_stdlib_job_and_io():
    for m in _imports(WORKER / "radar_worker" / "volume.py"):
        assert m.split(".")[0] in sys.stdlib_module_names or m in {"radar_worker.job", "radar_worker.io"}, m


def test_modal_app_takes_the_volume_helpers_from_radar_worker_volume():
    tree = ast.parse((WORKER / "modal_app.py").read_text(encoding="utf-8"))
    imported = {a.name for node in tree.body if isinstance(node, ast.ImportFrom) and node.module == "radar_worker.volume"
                for a in node.names}
    assert {"ref_kind", "volume_path", "volume_io"} <= imported
    defined = {n.name for n in tree.body if isinstance(n, ast.FunctionDef)}
    assert not defined & {"ref_kind", "volume_path", "volume_io"}


@pytest.fixture()
def no_network(monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError(f"network access during import: {args}")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)


def test_modal_app_imports_without_side_effects(monkeypatch, no_network):
    modal = pytest.importorskip("modal")
    monkeypatch.chdir(ROOT)
    monkeypatch.syspath_prepend(str(WORKER))
    monkeypatch.setenv("RADAR_GPU", "A10,L40S")
    for key in ("MODAL_TOKEN_ID", "MODAL_TOKEN_SECRET"):
        monkeypatch.delenv(key, raising=False)
    sys.modules.pop("modal_app", None)
    modal_app = importlib.import_module("modal_app")
    try:
        assert modal_app.RADAR_GPU == ["A10", "L40S"]
        assert isinstance(modal_app.app, modal.App)
        assert modal_app.app.name == "radar-desk"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            functions = modal_app.app.registered_functions
        assert "score" in functions
        assert modal_app.FUNCTION_OPTIONS["gpu"] == ["A10", "L40S"]
        # the function spec is private in modal 1.6.0; check it when it is there
        spec = getattr(functions["score"], "_spec_", None)
        if spec is not None:
            assert spec.gpus == ["A10", "L40S"]
            assert spec.cpu == 4.0 and spec.memory == 16384
            assert sorted(spec.volumes) == ["/data", "/weights"]
    finally:
        sys.modules.pop("modal_app", None)


@pytest.fixture()
def modal_app(monkeypatch, no_network):
    pytest.importorskip("modal")
    monkeypatch.chdir(ROOT)
    monkeypatch.syspath_prepend(str(WORKER))
    monkeypatch.setenv("MODAL_DATA_VOLUME", "radar-data-test")
    sys.modules.pop("modal_app", None)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield importlib.import_module("modal_app")
    sys.modules.pop("modal_app", None)


def test_data_volume_is_mounted_read_write_on_both_functions(modal_app):
    import modal

    assert modal_app.DATA_VOLUME == "radar-data-test" and modal_app.DATA_DIR == "/data"
    assert isinstance(modal_app.data_volume, modal.Volume)
    for name in ("score", "smoke_score"):
        spec = getattr(modal_app.app.registered_functions[name], "_spec_", None)
        if spec is not None:
            # the spec holds the unwrapped Volume, so compare by name; no mount options means read-write
            assert repr(spec.volumes["/data"]) == "modal.Volume.from_name('radar-data-test')", name


def test_ref_kind(modal_app):
    arts = {n: f"volume://jobs/j/{f}" for n, (f, _) in modal_app.ARTEFACTS.items()}
    assert modal_app.ref_kind("volume://scans/s/source.nii.gz", arts) == "volume"
    urls = {n: "https://b.example/x?sig=1" for n in arts}
    assert modal_app.ref_kind("https://b.example/s?sig=1", urls) == "url"
    assert modal_app.ref_kind("https://b.example/s?sig=1", arts) is None
    assert modal_app.ref_kind("volume://scans/s/source.nii.gz", {**arts, "log": "https://b.example/l"}) is None


@pytest.mark.parametrize("bad", ["volume://", "volume:///abs", "volume://a/../b", "volume://a//b",
                                 "volume://a\\b", "https://x/y"])
def test_volume_path_rejects_bad_refs(modal_app, tmp_path, bad):
    with pytest.raises(ValueError):
        modal_app.volume_path(bad, tmp_path)


def _volume_io(modal_app, root: Path, source_name: str = "source.nii.gz"):
    commits = []
    arts = {n: f"volume://jobs/j1/{f}" for n, (f, _) in modal_app.ARTEFACTS.items()}
    fetch, publish = modal_app.volume_io(f"volume://scans/s1/{source_name}", arts, root,
                                         lambda: commits.append(1))
    return fetch, publish, commits


def test_volume_fetch_reads_in_place_and_copies_only_to_rename(modal_app, tmp_path):
    import gzip

    root, work = tmp_path / "data", tmp_path / "work"
    work.mkdir()
    src = root / "scans" / "s1" / "source.nii.gz"
    src.parent.mkdir(parents=True)
    src.write_bytes(gzip.compress(b"nifti"))
    fetch, _, _ = _volume_io(modal_app, root)
    assert fetch(work, print) == src  # gzip under a .nii.gz name: read where it is
    src.write_bytes(b"plain nifti")  # a plain .nii uploaded under the .nii.gz key
    got = fetch(work, print)
    assert got == work / "source.nii" and got.read_bytes() == b"plain nifti"
    assert src.is_file()  # the Volume copy is never renamed
    assert modal_app.ensure_nifti_name(got) == got


def test_volume_fetch_missing_is_a_4xx(modal_app, tmp_path):
    from radar_worker.io import TransferError

    fetch, _, _ = _volume_io(modal_app, tmp_path / "data")
    with pytest.raises(TransferError) as info:
        fetch(tmp_path, print)
    assert info.value.status == 404


def test_volume_publish_writes_once_and_commits_after_the_fifth(modal_app, tmp_path):
    from radar_worker.io import TransferError

    root = tmp_path / "data"
    _, publish, commits = _volume_io(modal_app, root)
    art = tmp_path / "a.bin"
    art.write_bytes(b"artefact")
    names = list(modal_app.ARTEFACTS)
    for i, name in enumerate(names):
        publish(name, art, modal_app.ARTEFACTS[name][1])
        assert commits == ([1] if i == len(names) - 1 else [])
    for fname, _ in modal_app.ARTEFACTS.values():
        assert (root / "jobs" / "j1" / fname).read_bytes() == b"artefact"
    _, again, commits = _volume_io(modal_app, root)
    with pytest.raises(TransferError) as info:
        again("mask", art, "application/gzip")
    assert info.value.status == 409 and commits == []


def test_radar_gpu_single_value_and_default(monkeypatch, no_network):
    pytest.importorskip("modal")
    monkeypatch.chdir(ROOT)
    monkeypatch.syspath_prepend(str(WORKER))
    for value, expected in (("L4", "L4"), (None, "L4"), (" A100-40GB , L40S ", ["A100-40GB", "L40S"])):
        if value is None:
            monkeypatch.delenv("RADAR_GPU", raising=False)
        else:
            monkeypatch.setenv("RADAR_GPU", value)
        sys.modules.pop("modal_app", None)
        try:
            assert importlib.import_module("modal_app").RADAR_GPU == expected
        finally:
            sys.modules.pop("modal_app", None)


def _spike(monkeypatch, no_network):
    pytest.importorskip("modal")
    monkeypatch.syspath_prepend(str(ROOT / "worker"))
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    sys.modules.pop("modal_spike", None)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return importlib.import_module("modal_spike")


def test_spike_headroom_uses_reserved_memory(monkeypatch, no_network):
    spike = _spike(monkeypatch, no_network)
    total = 24 * 1024**3
    scan = {"path": "in/x/image1.nii.gz", "ok": True,
            "peak_allocated_bytes": int(0.60 * total), "peak_reserved_bytes": int(0.85 * total)}
    # 40% free on allocated but only 15% on reserved: the rule must say no.
    assert spike.headroom_ok({"total_bytes": total, "scans": [scan]}, scan["path"]) is False
    scan["peak_reserved_bytes"] = int(0.65 * total)
    assert spike.headroom_ok({"total_bytes": total, "scans": [scan]}, scan["path"]) is True
    assert spike.headroom_ok({"total_bytes": total, "scans": []}, scan["path"]) is False


def test_spike_fallback_has_more_memory_than_the_pick(monkeypatch, no_network):
    spike = _spike(monkeypatch, no_network)
    assert spike.fallback_for("L4") == "L40S"  # A10 is also 24 GB, so it is no fallback
    assert spike.fallback_for("A10") == "L40S"
    assert spike.fallback_for("L40S") == "A100-80GB"
    assert spike.fallback_for("B200") is None
    for name, nxt in spike.FALLBACK.items():
        assert spike.GPU_MEMORY_GB[nxt] > spike.GPU_MEMORY_GB[name], (name, nxt)


def test_volume_path_maps_keys_under_the_mount(modal_app):
    assert modal_app.volume_path("volume://jobs/j1/mask.nii.gz", "/data") == Path("/data/jobs/j1/mask.nii.gz")
    assert modal_app.volume_path("volume://scans/s1/source.nii.gz", modal_app.DATA_DIR) == Path(
        "/data/scans/s1/source.nii.gz")


class _NoReload:
    def reload(self):
        raise AssertionError("the Volume must not be reloaded for a rejected call")

    commit = reload


def _raw_score(modal_app, monkeypatch):
    raw = getattr(modal_app.score, "_raw_f_", None)  # private in modal 1.6.0; skip when it moves
    if raw is None:
        pytest.skip("the undecorated score function is not reachable in this modal version")
    monkeypatch.setattr(modal_app, "data_volume", _NoReload())
    return raw


def test_score_rejects_mixed_and_bad_volume_refs(modal_app, monkeypatch):
    score = _raw_score(modal_app, monkeypatch)
    arts = {n: f"volume://jobs/j1/{f}" for n, (f, _) in modal_app.ARTEFACTS.items()}
    out = score("j1", "volume://scans/s1/source.nii.gz", {**arts, "log": "https://b.example/l?sig=1"})
    assert out["ok"] is False and out["error"]["class"] == "input_error" and "mixed" in out["error"]["message"]
    out = score("j1", "https://b.example/s?sig=1", arts)
    assert out["error"]["class"] == "input_error" and "mixed" in out["error"]["message"]
    out = score("j1", "volume://scans/../etc/passwd", arts)
    assert out["error"]["class"] == "input_error" and "bad volume key" in out["error"]["message"]


def test_existing_artefact_is_artefact_exists_not_input_error(modal_app, monkeypatch, tmp_path):
    """run_job with inference stubbed out: a key already on the Volume fails as artefact_exists."""
    import gzip
    from types import SimpleNamespace

    from radar_worker import geometry, infer

    loaded = SimpleNamespace(test_items=["k1"], csv_header=lambda: ["file_name", "k1"])
    scored = {"ok": True, "file_name": "source.nii.gz", "findings": [{"key": "k1", "organ": "O", "finding": "F", "prob": 0.5}],
              "organs_scored": [], "organs_not_found": [], "organ_stats": {}, "trace": {},
              "timings": {"infer_s": 1.0, "postprocess_s": 0.1}, "mask": None, "affine": None}

    def write_mask(mask, affine, path):
        Path(path).write_bytes(b"mask")
        return Path(path)

    monkeypatch.setattr(modal_app, "weights_state", lambda log: {"ok": True, "checkpoint_sha256": "x"})
    monkeypatch.setattr(modal_app, "versions", lambda sha: {})
    monkeypatch.setattr(infer, "load_model", lambda d, device=None: loaded)
    monkeypatch.setattr(infer, "score_file", lambda p, m, log: dict(scored))
    monkeypatch.setattr(geometry, "write_mask", write_mask)

    root = tmp_path / "data"
    src = root / "scans" / "s1" / "source.nii.gz"
    src.parent.mkdir(parents=True)
    src.write_bytes(gzip.compress(b"nifti"))
    taken = root / "jobs" / "j1" / "mask.nii.gz"
    taken.parent.mkdir(parents=True)
    taken.write_bytes(b"from the first attempt")
    fetch, publish, commits = _volume_io(modal_app, root)
    out = modal_app.run_job("j1", fetch, publish)
    assert out["ok"] is False and out["error"]["class"] == "artefact_exists"
    assert "volume://jobs/j1/mask.nii.gz" in out["error"]["message"]
    assert taken.read_bytes() == b"from the first attempt" and commits == []


# ---------------------------------------------------------------- devices (no torch needed)


class _FakeDevice:
    def __init__(self, kind):
        self.type = kind

    def __str__(self):
        return self.type


def _fake_torch(cuda: bool, mps: bool):
    from types import SimpleNamespace

    return SimpleNamespace(
        device=_FakeDevice,
        cuda=SimpleNamespace(is_available=lambda: cuda, get_device_name=lambda d=None: "NVIDIA L4"),
        backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: mps)),
    )


@pytest.fixture()
def infer_mod(monkeypatch):
    monkeypatch.syspath_prepend(str(WORKER))
    monkeypatch.delenv("RADAR_DEVICE", raising=False)
    from radar_worker import infer

    return infer


@pytest.mark.parametrize(("cuda", "mps", "want"), [
    (True, True, "cuda"), (True, False, "cuda"), (False, True, "mps"), (False, False, "cpu"),
])
def test_auto_device_prefers_cuda_then_mps_then_cpu(infer_mod, monkeypatch, cuda, mps, want):
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(cuda, mps))
    assert infer_mod.resolve_device().type == want  # RADAR_DEVICE unset means auto
    assert infer_mod.resolve_device("auto").type == want
    monkeypatch.setenv("RADAR_DEVICE", " AUTO ")
    assert infer_mod.resolve_device().type == want


def test_explicit_device_and_env(infer_mod, monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(cuda=False, mps=True))
    assert infer_mod.resolve_device("cpu").type == "cpu"
    assert infer_mod.resolve_device("mps").type == "mps"
    monkeypatch.setenv("RADAR_DEVICE", "cpu")
    assert infer_mod.resolve_device().type == "cpu"
    assert infer_mod.resolve_device("mps").type == "mps"  # an argument wins over the env
    dev = _FakeDevice("mps")
    assert infer_mod.resolve_device(dev) is dev


@pytest.mark.parametrize(("name", "cuda", "mps"), [("cuda", False, True), ("mps", True, False), ("tpu", True, True)])
def test_unavailable_or_unknown_device_raises(infer_mod, monkeypatch, name, cuda, mps):
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(cuda, mps))
    with pytest.raises(ValueError, match=name):
        infer_mod.resolve_device(name)


def test_device_name_per_device(infer_mod, monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(cuda=True, mps=True))
    assert infer_mod.device_name(_FakeDevice("cuda")) == "NVIDIA L4"
    assert infer_mod.device_name(_FakeDevice("mps")) == "Apple MPS"
    assert infer_mod.device_name(_FakeDevice("cpu")) == "cpu"


def test_peak_rss_units(infer_mod):
    assert infer_mod.peak_rss_bytes(2048, "darwin") == 2048
    assert infer_mod.peak_rss_bytes(2048, "linux") == 2048 * 1024


def _center_crop_on_cuda(shape, crop_size):
    """Upstream center_crop's arithmetic for an empty mask, with CUDA's inf -> INT64_MAX."""
    big = 2**63 - 1
    lo = {"x": big, "y": big, "z": big}
    hi = {"x": 0, "y": 0, "z": 0}
    d, h, w = shape
    out = []
    for axis, n, c in (("z", d, crop_size[0]), ("y", h, crop_size[1]), ("x", w, crop_size[2])):
        size = max(c, hi[axis] - lo[axis])
        centre = (lo[axis] + hi[axis]) // 2
        start = max(0, centre - size // 2)
        end = min(n, start + size)
        if end - start < size:
            start = max(0, end - size)
        out.append([start, end])
    return out


@pytest.mark.parametrize("shape", [(96, 256, 384), (128, 288, 416), (192, 512, 512), (64, 224, 352)])
def test_far_corner_box_matches_cuda_arithmetic(infer_mod, shape):
    assert infer_mod.far_corner_box((1, 1, *shape)) == _center_crop_on_cuda(shape, infer_mod.ROI_SIZE)


def test_modal_loads_the_model_on_cuda_explicitly(modal_app, monkeypatch):
    """A container with broken CUDA fails at load instead of scoring on another device."""
    from radar_worker import infer

    asked = []

    class Stop(Exception):
        pass

    def load_model(weights_dir, device=None):
        asked.append(device)
        raise Stop

    monkeypatch.setattr(modal_app, "weights_state", lambda log: {"ok": True, "checkpoint_sha256": "x"})
    monkeypatch.setattr(infer, "load_model", load_model)
    with pytest.raises(Stop):
        modal_app.run_job("j1", lambda work, log: None, lambda *a: None)
    assert asked == ["cuda"]
    assert "env" not in modal_app.FUNCTION_OPTIONS


@pytest.mark.parametrize("name", ["parity.py", "modal_spike.py"])
def test_modal_scripts_load_the_model_on_cuda(name):
    tree = ast.parse((ROOT / "scripts" / name).read_text(encoding="utf-8"))
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Attribute) and n.func.attr == "load_model"]
    assert calls, name
    for call in calls:
        devices = [k.value.value for k in call.keywords if k.arg == "device" and isinstance(k.value, ast.Constant)]
        assert devices == ["cuda"], (name, ast.unparse(call))


def test_score_local_device_choices(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    monkeypatch.syspath_prepend(str(WORKER))
    sys.modules.pop("score_local", None)
    score_local = importlib.import_module("score_local")
    try:
        args = score_local.parse_args(["--path", "a.nii.gz", "--weights", "w", "--out", "o.json"])
        assert args.device == "auto"
        args = score_local.parse_args(["--device", "mps", "--path", "a", "--weights", "w", "--out", "o"])
        assert args.device == "mps"
        with pytest.raises(SystemExit):
            score_local.parse_args(["--device", "tpu", "--path", "a", "--weights", "w", "--out", "o"])
    finally:
        sys.modules.pop("score_local", None)
