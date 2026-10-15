from __future__ import annotations

import json
from pathlib import Path
import pytest

from src.pose_control.qwen21_full.prompt_study.protocol import P0_PROMPT
from src.pose_control.qwen21_full.prompt_study.prompts import (
    request_roles,
    parse_answer,
    compile_prompt,
    make_p0_record,
    PromptBank,
)


def test_pg_never_requests_B_or_manual_caption():
    row = dict(
        source_rgb="a.png",
        target_rgb="SECRET_B.png",
        target_depth_vis="d.png",
        target_normal="n.png",
        filtered_dwpose="p.png",
        action_description="SECRET_ACTION",
    )
    req = request_roles(row, "pg")
    assert req == [
        ("source_A", "a.png"),
        ("target_depth", "d.png"),
        ("target_normal", "n.png"),
        ("target_dwpose", "p.png"),
    ]
    assert "SECRET" not in str(req)


def test_compiled_text_has_no_unprovided_image_tokens():
    raw = "<image1>: Raise both arms. Follow <image2>. Preserve clothing."
    result = compile_prompt(raw, ("source_A", "target_B"))
    assert "<image" not in result and "Raise both arms" in result
    with pytest.raises(ValueError, match="image"):
        compile_prompt("Use <image9>", ("source_A", "target_B"))
    assert (
        parse_answer('thinking</think>{"rewritten_prompt":"Raise both arms"}')[
            "rewritten_prompt"
        ]
        == "Raise both arms"
    )


def test_parse_answer_fences_and_empty():
    fenced = "```json\n{\"rewritten_prompt\": \"Bend forward slightly.\"}\n```"
    assert parse_answer(fenced)["rewritten_prompt"] == "Bend forward slightly."

    with pytest.raises(ValueError, match="Missing nonempty"):
        parse_answer('{"rewritten_prompt": ""}')


def test_prompt_bank_loads_valid_and_rejects_invalid(tmp_path: Path):
    manifest = tmp_path / "train.jsonl"
    r1 = make_p0_record("pair_0", "train")
    r2 = make_p0_record("pair_1", "train")
    manifest.write_text(
        json.dumps(r1) + "\n" + json.dumps(r2) + "\n",
        encoding="utf-8",
    )

    bank = PromptBank(manifest, "p0", ["pair_0", "pair_1"])
    assert bank.prompt_for("pair_0") == P0_PROMPT
    assert bank.prompt_for("pair_1") == P0_PROMPT

    # Reject incomplete set
    with pytest.raises(ValueError, match="cover the locked split"):
        PromptBank(manifest, "p0", ["pair_0", "pair_1", "pair_2"])

    # Reject corrupt hash
    corrupt = make_p0_record("pair_0", "train")
    corrupt["compiled_prompt_sha256"] = "wrong_hash"
    bad_manifest = tmp_path / "bad.jsonl"
    bad_manifest.write_text(json.dumps(corrupt) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Prompt hash differs"):
        PromptBank(bad_manifest, "p0", ["pair_0"])


def test_parse_answer_with_surrounding_conversational_text():
    raw = (
        "Here is the result based on your instructions:\n"
        "```json\n"
        '{"rewritten_prompt": "Cross both arms over the chest."}\n'
        "```\n"
        "Let me know if you need any adjustments!"
    )
    assert parse_answer(raw)["rewritten_prompt"] == "Cross both arms over the chest."


def test_compile_prompt_handles_image_naming_variants():
    raw = "Keep identity from Image_1. Change pose to match <image_2>."
    compiled = compile_prompt(raw, ("source_A", "target_B"))
    assert "Image_1" not in compiled
    assert "<image_2>" not in compiled
    assert "the source image" in compiled
    assert "the target pose described in this instruction" in compiled


def test_request_roles_geometry_fallback(tmp_path: Path):
    geom_dir = tmp_path / "geometry/pair_99"
    geom_dir.mkdir(parents=True, exist_ok=True)
    filt_p = geom_dir / "dwpose_filtered.png"
    filt_p.write_bytes(b"PNG")

    row = {
        "pair_id": "pair_99",
        "source_rgb": "a.png",
        "target_depth_vis": "d.png",
        "target_normal": "n.png",
    }
    # Should find filtered dwpose under geometry directory
    roles = request_roles(row, "pg", assets_dir=tmp_path / "geometry")
    assert any(role == "target_dwpose" and path == str(filt_p) for role, path in roles)


def test_request_roles_rejects_unfiltered_skeleton():
    row = dict(pair_id="p1", source_rgb="a.png", target_depth_vis="d.png",
               target_normal="n.png", target_dwpose="unfiltered.png",
               target_dwpose_png="also_unfiltered.png")
    with pytest.raises(KeyError, match="target_dwpose"):
        request_roles(row, "pg")


def test_pe_runner_requires_model_and_official_system(tmp_path):
    from scripts.precompute_qwen21_pei2i_prompts import PEModelRunner
    with pytest.raises((ValueError, FileNotFoundError), match="model"):
        PEModelRunner(None)
    with pytest.raises(FileNotFoundError, match="system_prompt"):
        PEModelRunner(tmp_path)


def test_p0_precompute_never_loads_pe(tmp_path, monkeypatch):
    import scripts.precompute_qwen21_pei2i_prompts as pe
    data = tmp_path / "data"
    data.mkdir()
    (data / "train_pairs.jsonl").write_text('{"pair_id":"p1"}\n')
    def forbidden(*args, **kwargs):
        pytest.fail("P0 attempted to load PE")
    monkeypatch.setattr(pe, "PEModelRunner", forbidden)
    result = pe.process_split_prompts(tmp_path, "train", ["p0"])
    assert result["counts"]["p0"] == 1


def test_prompt_bank_rejects_synthetic_pe_provenance(tmp_path):
    from src.pose_control.qwen21_full.prompt_study.protocol import PROMPT_ROLES
    row = make_p0_record("p1", "train")
    row.update(protocol="pb", input_roles=list(PROMPT_ROLES["pb"]),
               inputs=[dict(role=role, path="fake.png", sha256="f" * 64,
                            size=[10, 10], asset_relative_path="prompts/input_assets/fake.png")
                       for role in PROMPT_ROLES["pb"]])
    path = tmp_path / "train.jsonl"
    path.write_text(json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="provenance"):
        PromptBank(path, "pb", ["p1"])


@pytest.fixture
def pe_run(tmp_path):
    import torch
    from PIL import Image
    from scripts.precompute_qwen21_pei2i_prompts import PEModelRunner
    run = tmp_path / "run"
    data = run / "data"
    data.mkdir(parents=True)
    a, b = data / "a.png", data / "b.png"
    Image.new("RGB", (16, 24), "red").save(a)
    Image.new("RGB", (16, 24), "blue").save(b)
    rows = [dict(pair_id="p1", source_rgb=str(a), target_rgb=str(b))]
    (data / "train_pairs.jsonl").write_text(json.dumps(rows[0]) + "\n")
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "system_prompt.txt").write_text("Official test system text.")
    (model_dir / "config.json").write_text('{"_commit_hash":"test-revision"}')
    (model_dir / "model.safetensors").write_bytes(b"test weight bytes")

    class Batch(dict):
        def to(self, device):
            return self

    class Processor:
        template_calls = []
        def apply_chat_template(self, messages, **kwargs):
            self.template_calls.append((messages, kwargs))
            return "rendered prompt"
        def __call__(self, **kwargs):
            return Batch(input_ids=torch.tensor([[1, 2, 3]]))
        def decode(self, ids, **kwargs):
            return '<think>observing</think>{"rewritten_prompt":"Raise both arms. Keep Image 1 identity."}'

    class Model:
        calls = []
        def generate(self, **kwargs):
            self.calls.append(kwargs)
            return torch.tensor([[1, 2, 3, 20, 21, 22, 23, 24]])

    processor, model = Processor(), Model()
    runner = PEModelRunner(model_dir, device="cpu", processor=processor, model=model)
    return run, runner, processor, model


def test_pe_generation_uses_thinking_full_seeds_and_token_ids(pe_run):
    import random
    import numpy as np
    import torch
    _, runner, processor, model = pe_run
    random.seed(1); np.random.seed(1); torch.manual_seed(1)
    result = runner.generate("instruction", [], seed=43)
    values = (random.random(), np.random.rand(), torch.rand(1).item())
    runner.generate("instruction", [], seed=43)
    assert values == (random.random(), np.random.rand(), torch.rand(1).item())
    assert result.tokens == 5
    assert result.raw_output.startswith("<think>")
    messages, kwargs = processor.template_calls[0]
    assert kwargs["enable_thinking"] is True
    assert messages[0]["content"] == runner.system_prompt
    assert model.calls[0]["max_new_tokens"] == 24000


def test_prompt_precompute_loads_template_and_reuses_only_exact_inputs(pe_run):
    from PIL import Image
    import scripts.precompute_qwen21_pei2i_prompts as pe
    from src.pose_control.qwen21_full.prompt_study.protocol import sha256_file
    run, runner, _, model = pe_run
    prompts = run / "prompts"
    prompts.mkdir()
    template = pe.default_template()
    template["common"] = "Write a standalone precise pose edit."
    (prompts / "instruction_v1.json").write_text(json.dumps(template))
    pe.process_split_prompts(run, "train", ["pb"], runner=runner)
    records = pe.read_jsonl(prompts / "pb/train.jsonl")
    assert len(records) == 1
    first = records[0]
    assert first["tokens"] == 5
    assert first["instruction_sha256"] == sha256_file(prompts / "instruction_v1.json")
    assert not Path(first["raw_path"]).is_absolute()
    raw = json.loads((run / first["raw_path"]).read_text())
    assert raw["instruction"].startswith(template["common"])
    assert raw["system_prompt"] == runner.system_prompt
    PromptBank(prompts / "pb/train.jsonl", "pb", ["p1"])
    pe.process_split_prompts(run, "train", ["pb"], runner=runner)
    assert len(model.calls) == 1
    Image.new("RGB", (16, 24), "green").save(run / "data/b.png")
    pe.process_split_prompts(run, "train", ["pb"], runner=runner)
    changed = pe.read_jsonl(prompts / "pb/train.jsonl")
    assert len(changed) == 1
    assert changed[0]["request_sha256"] != first["request_sha256"]
    assert (run / first["raw_path"]).is_file(), "old raw evidence must be preserved"
    assert len(model.calls) == 2
    PromptBank(prompts / "pb/train.jsonl", "pb", ["p1"])
    template["common"] += " Keep the background."
    (prompts / "instruction_v1.json").write_text(json.dumps(template))
    with pytest.raises(ValueError, match="provenance"):
        PromptBank(prompts / "pb/train.jsonl", "pb", ["p1"])
    pe.process_split_prompts(run, "train", ["pb"], runner=runner)
    assert len(model.calls) == 3


def test_qa8_has_separate_unique_manifests(pe_run):
    import scripts.precompute_qwen21_pei2i_prompts as pe
    run, runner, _, model = pe_run
    for _ in range(2):
        pe.process_split_prompts(run, "train", ["pb"], qa8_mode=True, runner=runner)
    assert not (run / "prompts/pb/train.jsonl").exists()
    assert len(pe.read_jsonl(run / "prompts/qa8/pb/train.jsonl")) == 1
    assert len(model.calls) == 1
    pe.process_split_prompts(run, "train", ["pb"], runner=runner)
    assert len(pe.read_jsonl(run / "prompts/pb/train.jsonl")) == 1
    assert len(model.calls) == 1, "exact reviewed development signatures may be promoted"


def test_prompt_bank_rejects_tampered_raw_or_asset(pe_run):
    import scripts.precompute_qwen21_pei2i_prompts as pe
    run, runner, _, _ = pe_run
    pe.process_split_prompts(run, "train", ["pb"], runner=runner)
    manifest = run / "prompts/pb/train.jsonl"
    row = pe.read_jsonl(manifest)[0]
    asset = run / row["inputs"][0]["asset_relative_path"]
    original = asset.read_bytes()
    asset.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="provenance"):
        PromptBank(manifest, "pb", ["p1"])
    asset.write_bytes(original)
    raw = run / row["raw_path"]
    raw.write_text("{}")
    with pytest.raises(ValueError, match="provenance"):
        PromptBank(manifest, "pb", ["p1"])


def test_pe_load_errors_propagate(tmp_path, monkeypatch):
    import sys
    from types import SimpleNamespace
    from scripts.precompute_qwen21_pei2i_prompts import PEModelRunner
    (tmp_path / "system_prompt.txt").write_text("system")
    (tmp_path / "config.json").write_text("{}")
    (tmp_path / "model.safetensors").write_bytes(b"weights")
    def broken(*args, **kwargs):
        raise RuntimeError("model shard unreadable")
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(
        AutoProcessor=SimpleNamespace(from_pretrained=broken),
        AutoModelForImageTextToText=SimpleNamespace(from_pretrained=broken)))
    with pytest.raises(RuntimeError, match="model shard unreadable"):
        PEModelRunner(tmp_path)


def test_pe_cli_loads_model_once_across_splits(tmp_path, monkeypatch):
    import sys
    import scripts.precompute_qwen21_pei2i_prompts as pe
    (tmp_path / "data").mkdir()
    for split in ("train", "val"):
        (tmp_path / "data" / f"{split}_pairs.jsonl").write_text('{"pair_id":"p1"}\n')
    loaded = []
    shared = object()
    def load(model):
        loaded.append(model)
        return shared
    monkeypatch.setattr(pe, "PEModelRunner", load)
    calls = []
    def process(*args, **kwargs):
        calls.append(kwargs.get("runner"))
        return {}
    monkeypatch.setattr(pe, "process_split_prompts", process)
    monkeypatch.setattr(pe, "validate_qa8_review", lambda run: {"model_dir":str(pe.DEFAULT_PE_MODEL.resolve())}, raising=False)
    monkeypatch.setattr(sys, "argv", ["pe.py", "--run",str(tmp_path),"--splits","train","val","--protocols","pb"])
    pe.main()
    assert loaded == [pe.DEFAULT_PE_MODEL]
    assert calls == [shared, shared]


def test_failed_pe_generation_archives_error_and_never_completes(pe_run, monkeypatch):
    import scripts.precompute_qwen21_pei2i_prompts as pe
    run, runner, _, _ = pe_run
    def broken(*args, **kwargs):
        raise RuntimeError("GPU inference failed")
    monkeypatch.setattr(runner, "generate", broken)
    with pytest.raises(RuntimeError, match="GPU inference failed"):
        pe.process_split_prompts(run, "train", ["pb"], runner=runner)
    assert not (run / "prompts/pb/train.jsonl").is_file()
    failures = list((run / "prompts/attempts").rglob("attempt_0.json"))
    assert len(failures) == 1
    raw = json.loads(failures[0].read_text())
    assert raw["status"] == "failed" and raw["tokens"] == 0
    assert "GPU inference failed" in raw["error"]
    assert raw["inputs"][0]["sha256"]


def test_invalid_pe_answer_retries_fixed_seeds_and_preserves_raw(pe_run, monkeypatch):
    import scripts.precompute_qwen21_pei2i_prompts as pe
    run, runner, processor, model = pe_run
    outputs = iter(["not valid JSON", '{"rewritten_prompt":"Lower the left arm. Preserve identity."}'])
    monkeypatch.setattr(processor, "decode", lambda *args, **kwargs: next(outputs))
    pe.process_split_prompts(run, "train", ["pb"], runner=runner)
    record = pe.read_jsonl(run / "prompts/pb/train.jsonl")[0]
    assert record["attempt"] == 1 and record["seed"] == 43
    failures = list((run / "prompts/attempts").rglob("attempt_0.json"))
    assert len(failures) == 1
    assert json.loads(failures[0].read_text())["raw_output"] == "not valid JSON"
    assert json.loads(failures[0].read_text())["error"]
    assert len(model.calls) == 2


def test_prompt_bank_requires_raw_evidence_outside_named_run_directory(pe_run):
    import scripts.precompute_qwen21_pei2i_prompts as pe
    run, runner, _, _ = pe_run
    pe.process_split_prompts(run, "train", ["pb"], runner=runner)
    # A standalone JSONL must not silently bypass evidence checks.
    standalone = run.parent / "standalone.jsonl"
    standalone.write_text((run / "prompts/pb/train.jsonl").read_text())
    with pytest.raises(ValueError, match="provenance"):
        PromptBank(standalone, "pb", ["p1"])


def test_timing_records_real_tokens_failures_and_retry_count(pe_run, monkeypatch):
    import scripts.precompute_qwen21_pei2i_prompts as pe
    run, runner, processor, _ = pe_run
    outputs = iter(["invalid", '{"rewritten_prompt":"Raise the right arm."}'])
    monkeypatch.setattr(processor, "decode", lambda *args, **kwargs: next(outputs))
    pe.process_split_prompts(run, "train", ["pb"], runner=runner)
    timing_file = run / "prompts/timing.json"
    assert timing_file.is_file(), "record actual first requests for runtime estimation"
    timing = json.loads(timing_file.read_text())
    assert timing["attempt_count"] == 2
    assert timing["failed_attempt_count"] == 1
    assert timing["retry_attempt_count"] == 1
    assert timing["generated_tokens"] == 10
    assert len(timing["first20"]) == 2


def test_p0_provenance_validator_rejects_pe_fields():
    from src.pose_control.qwen21_full.prompt_study.prompts import validate_prompt_record
    row = make_p0_record("p1", "train")
    row["model_dir"] = "/model"
    with pytest.raises(ValueError, match="P0 provenance"):
        validate_prompt_record(row)


def test_pe_rewrite_rejects_changed_locked_model(pe_run, monkeypatch):
    import scripts.precompute_qwen21_pei2i_prompts as pe
    run, runner, _, model = pe_run
    (run / "data/lock.json").write_text("{}")
    # Explicit isolated metadata injection avoids fabricating a valid full study lock.
    monkeypatch.setattr(pe, "load_study_lock", lambda *args, **kwargs: {
        "artifacts": {"pe_model": {"path": str(runner.model_dir),
                       "files": {"model.safetensors": {"sha256": "0" * 64}}}}}, raising=False)
    with pytest.raises(ValueError, match="PE model provenance"):
        pe.process_split_prompts(run, "train", ["pb"], runner=runner)
    assert model.calls == []


def test_formal_pe_cli_requires_review_before_model_load(tmp_path, monkeypatch):
    import sys
    import scripts.precompute_qwen21_pei2i_prompts as pe
    (tmp_path / "data").mkdir()
    (tmp_path / "data/train_pairs.jsonl").write_text('{"pair_id":"p1"}\n')
    loaded = []
    def forbidden(*args, **kwargs):
        loaded.append(True)
        pytest.fail("PE loaded before QA8 review was verified")
    monkeypatch.setattr(pe, "PEModelRunner", forbidden)
    monkeypatch.setattr(sys, "argv", ["pe.py", "--run", str(tmp_path), "--splits", "train", "--protocols", "pb"])
    with pytest.raises(ValueError, match="QA8"):
        pe.main()
    assert loaded == []


@pytest.fixture
def reviewed_qa8(pe_run, monkeypatch):
    import scripts.precompute_qwen21_pei2i_prompts as pe
    from src.pose_control.qwen21_full.prompt_study.protocol import directory_artifact
    run, runner, _, _ = pe_run
    original = pe.read_jsonl(run / "data/train_pairs.jsonl")[0]
    rows = [{**original, "pair_id": f"p{i}", "target_depth_vis":original["target_rgb"],
             "target_normal":original["target_rgb"], "filtered_dwpose":original["target_rgb"]}
            for i in range(8)]
    pe.write_jsonl(run / "data/train_pairs.jsonl", rows)
    artifact = directory_artifact(runner.model_dir, revision=runner.revision)
    lock = {"artifacts":{"pe_model": artifact}, "splits":{"train":{"path":"train_pairs.jsonl"}}}
    (run / "data/lock.json").write_text("{}")
    monkeypatch.setattr(pe, "load_study_lock", lambda *args, **kwargs: lock)
    pe.process_split_prompts(run, "train", ["pg", "pb", "pbg"], qa8_mode=True, runner=runner)
    assert hasattr(pe, "qa8_review_provenance"), "review needs a concrete immutable provenance payload"
    payload = pe.qa8_review_provenance(run)
    payload.update(reviewed=True, notes=["Inspected all 24 replies against their permitted images."])
    payload["review_signature_sha256"] = pe.request_sha256(payload)
    pe.atomic_json(run / "prompts/qa8.json", payload)
    return run, payload, lock


def test_qa8_review_accepts_exact_completed_review(reviewed_qa8):
    import scripts.precompute_qwen21_pei2i_prompts as pe
    run, payload, _ = reviewed_qa8
    assert pe.validate_qa8_review(run) == payload


@pytest.mark.parametrize("field", ["reviewed", "instruction_sha256", "qa_manifest_sha256", "compiler_sha256",
                                    "model_dir", "model_manifest_sha256", "request_sha256", "notes", "review_signature_sha256"])
def test_qa8_review_rejects_stale_or_incomplete_review(reviewed_qa8, field):
    import copy
    import scripts.precompute_qwen21_pei2i_prompts as pe
    run, original, _ = reviewed_qa8
    payload = copy.deepcopy(original)
    if field == "reviewed":
        payload[field] = False
    elif field == "notes":
        payload[field] = []
    elif field == "request_sha256":
        payload[field]["pg"]["p0"] = "0" * 64
    else:
        payload[field] = "wrong"
    if field != "review_signature_sha256":
        payload.pop("review_signature_sha256")
        payload["review_signature_sha256"] = pe.request_sha256(payload)
    pe.atomic_json(run / "prompts/qa8.json", payload)
    with pytest.raises(ValueError, match="QA8"):
        pe.validate_qa8_review(run)


@pytest.mark.parametrize("change", ["bank_bytes", "locked_order", "locked_model", "input_asset", "template", "missing_protocol"])
def test_qa8_review_binds_locked_first8_and_actual_bank_bytes(reviewed_qa8, change):
    import scripts.precompute_qwen21_pei2i_prompts as pe
    run, _, lock = reviewed_qa8
    path = run / "prompts/qa8/pb/train.jsonl"
    if change == "bank_bytes":
        rows = pe.read_jsonl(path)
        rows[0]["seconds"] += 1.0
        pe.write_jsonl(path, rows)
    elif change == "locked_order":
        pairs = pe.read_jsonl(run / "data/train_pairs.jsonl")
        pe.write_jsonl(run / "data/train_pairs.jsonl", list(reversed(pairs)))
    elif change == "locked_model":
        lock["artifacts"]["pe_model"]["files"]["config.json"]["sha256"] = "0" * 64
    elif change == "input_asset":
        from PIL import Image
        Image.new("RGB", (16, 24), "yellow").save(run / "data/a.png")
    elif change == "template":
        template = json.loads((run / "prompts/instruction_v1.json").read_text())
        template["common"] = "A different pose instruction."
        pe.atomic_json(run / "prompts/instruction_v1.json", template)
    else:
        (run / "prompts/qa8/pbg/train.jsonl").unlink()
    with pytest.raises(ValueError, match="QA8"):
        pe.validate_qa8_review(run)


def test_pe_qa_and_p0_modes_are_review_exempt(tmp_path, monkeypatch):
    import sys
    import scripts.precompute_qwen21_pei2i_prompts as pe
    (tmp_path / "data").mkdir()
    (tmp_path / "data/train_pairs.jsonl").write_text('{"pair_id":"p1"}\n')
    monkeypatch.setattr(pe, "validate_qa8_review", lambda run: pytest.fail("Exempt mode invoked review gate"), raising=False)
    monkeypatch.setattr(pe, "PEModelRunner", lambda model: object())
    monkeypatch.setattr(pe, "process_split_prompts", lambda *args, **kwargs: {})
    for args in (["--qa8"], ["--splits", "train", "--protocols", "p0"]):
        monkeypatch.setattr(sys, "argv", ["pe.py", "--run", str(tmp_path), *args])
        pe.main()

