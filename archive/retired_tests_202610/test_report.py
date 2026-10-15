import csv
import io
import json
import struct
import tempfile
import unittest
from pathlib import Path

from vram_lab.report import aggregate, build_report, timeline_phases


class ReportTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def run_fixture(self, name, config=None, summary=None):
        directory = self.root / "runs" / name
        directory.mkdir(parents=True)
        (directory / "config.json").write_text(json.dumps(config or {"id": name}), encoding="utf-8")
        if summary is not None:
            (directory / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
        return directory

    def test_missing_summary_and_failure_are_visible_without_invented_metrics(self):
        self.run_fixture("missing")
        self.run_fixture("oom", summary={"status": "OOM", "peak_allocated_gib": 14.2})
        report = build_report(self.root)
        self.assertEqual(len(report["runs"]), 2)
        missing, oom = report["runs"]
        self.assertEqual(missing["status"], "NOT_RUN")
        self.assertFalse(missing["summary_present"])
        self.assertIsNone(missing["wall_seconds"])
        self.assertTrue(all(v is None for v in missing["scores"].values()))
        self.assertEqual(oom["status"], "OOM")
        self.assertEqual(oom["peak_allocated_gib"], 14.2)
        self.assertEqual(report["aggregate"]["counts"], {"NOT_RUN": 1, "OOM": 1})
        self.assertIsNone(report["aggregate"]["pass_metrics"]["wall_seconds"]["mean"])

    def test_real_values_scoring_overrides_and_aggregation(self):
        self.run_fixture("pass1", summary={"status": "PASS", "wall_seconds": 10, "scores": {"identity": 2, "action": 1}})
        self.run_fixture("pass2", summary={"status": "PASS", "wall_seconds": 20, "peak_nvml_gib": "bad", "scores": {"identity": None, "action": 0}})
        self.run_fixture("fail", summary={"status": "OOM", "wall_seconds": 90, "scores": {"action": 2}})
        (self.root / "scores.json").write_text(json.dumps({"pass1": {"identity": None, "action": 2, "added_person": False}}))
        report = build_report(self.root)
        agg = aggregate(report["runs"])
        self.assertEqual(agg["pass_metrics"]["wall_seconds"], {"mean": 15, "n": 2})
        self.assertEqual(agg["pass_scores"]["action"], {"mean": 1, "n": 2})
        self.assertEqual(agg["pass_scores"]["identity"], {"mean": None, "n": 0})
        first = next(r for r in report["runs"] if r["id"] == "pass1")
        self.assertFalse(first["scores"]["added_person"])
        rows = list(csv.DictReader(io.StringIO((self.root / "summary.csv").read_text(encoding="utf-8-sig"))))
        self.assertEqual(len(rows), 3)
        self.assertEqual(next(r for r in rows if r["id"] == "pass2")["peak_nvml_gib"], "")

    def test_html_payload_escaping_and_evidence_urls(self):
        malicious = '</script><img src=x onerror="alert(1)"> & 测试'
        directory = self.run_fixture("space & run", {"id": "=FORMULA()", "prompt": malicious}, {"status": "PASS"})
        (directory / "input_model.png").write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00\x00\x00\rIHDR" + struct.pack(">II", 128, 256))
        result = build_report(self.root)
        html = (self.root / "index.html").read_text(encoding="utf-8")
        self.assertNotIn(malicious, html)
        self.assertIn("\\u003c/script\\u003e", html)
        self.assertEqual(json.loads((self.root / "results.json").read_text(encoding="utf-8"))["runs"][0]["config"]["prompt"], malicious)
        self.assertEqual(result["runs"][0]["images"]["input_model"]["dimensions"], {"width": 128, "height": 256})
        self.assertEqual(result["runs"][0]["evidence"]["input_model.png"], "runs/space%20%26%20run/input_model.png")
        rows = list(csv.DictReader(io.StringIO((self.root / "summary.csv").read_text(encoding="utf-8-sig"))))
        self.assertEqual(rows[0]["id"], "'=FORMULA()")

    def test_empty_corrupt_and_partial_files_are_tolerated(self):
        self.assertEqual(build_report(self.root)["aggregate"]["total"], 0)
        directory = self.run_fixture("broken", summary={"status": "nonsense"})
        (directory / "events.jsonl").write_text('{bad}\n{"stage":"load","event":"end","elapsed_seconds":1}\n[]\n')
        (directory / "gpu_samples.csv").write_text('elapsed_seconds,used_gib,stage\n1,3.5,load\n2,NaN,load\n')
        report = build_report(self.root)
        run = report["runs"][0]
        self.assertEqual(run["status"], "INVALID")
        self.assertEqual(len(run["events"]), 1)
        self.assertEqual(run["gpu_samples"][0]["used_gib"], 3.5)
        self.assertIsNone(run["gpu_samples"][1]["used_gib"])

    def test_core_training_and_control_metadata_contract(self):
        directory = self.run_fixture("train", {"id": "train", "control_injected": True}, {
            "status": "PASS", "step_p50_seconds": 1.25, "step_p95_seconds": 1.7,
            "adapter_parameters": 1234, "validation_passed": False,
        })
        metadata = {"source": "artificial skeleton", "aligned_to_photo": False}
        (directory / "control_metadata.json").write_text(json.dumps(metadata))
        (directory / "source_version.json").write_text(json.dumps({"worker_sha256": "abc"}))
        run = build_report(self.root)["runs"][0]
        self.assertEqual(run["control_metadata"], metadata)
        self.assertEqual(run["source_version"]["worker_sha256"], "abc")
        self.assertEqual(run["step_p50_seconds"], 1.25)
        self.assertEqual(run["adapter_parameters"], 1234)
        self.assertFalse(run["validation_passed"])

    def test_stable_training_task_evidence_remains_aligned_and_separate(self):
        config = {"id": "stable", "kind": "train", "group": "stable", "task": "S0",
                  "training_tasks": ["S0", "D0"],
                  "task_evidence": {"S0": "task_evidence/S0", "D0": "task_evidence/D0", "unsafe": "../../"}}
        directory = self.run_fixture("stable", config, {"status": "PASS"})
        for task, width in (("S0", 320), ("D0", 640)):
            task_dir = directory / "task_evidence" / task
            task_dir.mkdir(parents=True)
            png = b"\x89PNG\r\n\x1a\n" + b"\x00\x00\x00\rIHDR" + struct.pack(">II", width, 240)
            for name in ("input_original", "input_model", "normal", "depth", "skeleton", "contact"):
                (task_dir / (name + ".png")).write_bytes(png)
            (task_dir / "control_metadata.json").write_text(json.dumps({"source": "artificial " + task, "aligned_to_photo": False}))
        run = build_report(self.root)["runs"][0]
        self.assertEqual(run["config"]["task"], "S0")
        self.assertEqual(set(run["task_evidence"]), {"S0", "D0"})
        self.assertNotIn("input_model", run["images"])
        for task, width in (("S0", 320), ("D0", 640)):
            evidence = run["task_evidence"][task]
            self.assertEqual(evidence["images"]["input_model"]["dimensions"]["width"], width)
            self.assertEqual(evidence["control_metadata"]["source"], "artificial " + task)
            self.assertEqual(evidence["images"]["normal"]["url"], f"runs/stable/task_evidence/{task}/normal.png")

    def test_capacity_is_pending_until_written_and_preserves_finalized_values(self):
        report = build_report(self.root)
        self.assertIsNone(report["capacity"])
        self.assertEqual(report["root_evidence"], {})
        capacity = [{"resolution": 512, "stable_peak_nvml_gib": 10.5,
                     "lifecycle_peak_nvml_gib": 18.5, "suggested_capacity_gib": 21,
                     "encoding": "cache", "checkpointing": True}]
        (self.root / "capacity.json").write_text(json.dumps(capacity))
        (self.root / "MEMORY_REPORT.md").write_text("Test-only capacity evidence")
        report = build_report(self.root)
        self.assertEqual(report["capacity"], capacity)
        self.assertEqual(report["root_evidence"], {"capacity.json": "capacity.json", "MEMORY_REPORT.md": "MEMORY_REPORT.md"})

    def test_ai_score_provenance_survives_merge_and_exports(self):
        self.run_fixture("reviewed", summary={"status": "PASS", "scores": {"identity": 1, "action": 0}})
        reviewer = "Codex visual review; not a human benchmark"
        note = "AI 初评；身份待人工确认。\n保留复核依据 </script>"
        (self.root / "scores.json").write_text(json.dumps({"reviewed": {
            "scores": {"action": 2, "identity": None}, "reviewer": reviewer, "note": note,
        }}), encoding="utf-8")
        result = build_report(self.root)
        scores = result["runs"][0]["scores"]
        self.assertEqual(scores["reviewer"], reviewer)
        self.assertEqual(scores["note"], note)
        self.assertEqual(scores["action"], 2)
        self.assertIsNone(scores["identity"])
        exported = json.loads((self.root / "results.json").read_text(encoding="utf-8"))["runs"][0]["scores"]
        self.assertEqual(exported["reviewer"], reviewer)
        self.assertEqual(exported["note"], note)
        row = next(csv.DictReader(io.StringIO((self.root / "summary.csv").read_text(encoding="utf-8-sig"))))
        self.assertEqual(row["reviewer"], reviewer)
        self.assertEqual(row["note"], note)

    def test_hundred_training_updates_merge_without_losing_evidence(self):
        directory = self.run_fixture("stable_timeline", {"id": "stable_timeline", "kind": "train"}, {"status": "PASS"})
        events = []
        for i in range(100):
            stage = "first_update" if i == 0 else "warmup" if i < 4 else "steady_train"
            events.extend([{"stage": stage, "event": "start", "elapsed_seconds": i * 2, "allocated_gib": 3},
                           {"stage": stage, "event": "end", "elapsed_seconds": i * 2 + 1, "allocated_gib": 4}])
        before = json.dumps(events)
        (directory / "events.jsonl").write_text("\n".join(json.dumps(event) for event in events))
        samples = "elapsed_seconds,used_gib,stage\n" + "\n".join(f"{i},5,steady_train" for i in range(200))
        (directory / "gpu_samples.csv").write_text(samples)
        result = build_report(self.root)["runs"][0]
        phases = result["timeline_phases"]
        self.assertEqual([(phase["stage"], phase["count"]) for phase in phases],
                         [("first_update", 1), ("warmup", 3), ("steady_train", 96)])
        self.assertEqual(phases[-1]["start_seconds"], 8)
        self.assertEqual(phases[-1]["end_seconds"], 199)
        self.assertTrue(phases[-1]["closed"])
        self.assertEqual(result["events"], events)
        self.assertEqual(json.dumps(events), before)
        self.assertEqual(len(result["gpu_samples"]), 200)

    def test_phase_reentry_is_separate_and_missing_end_is_not_invented(self):
        events = [{"stage": "steady_train", "event": "start", "elapsed_seconds": 0},
                  {"stage": "steady_train", "event": "end", "elapsed_seconds": 1},
                  {"stage": "validation", "event": "start", "elapsed_seconds": 2},
                  {"stage": "validation", "event": "end", "elapsed_seconds": 3},
                  {"stage": "steady_train", "event": "start", "elapsed_seconds": 4}]
        phases = timeline_phases(events)
        self.assertEqual([phase["stage"] for phase in phases], ["steady_train", "validation", "steady_train"])
        self.assertFalse(phases[-1]["closed"])
        self.assertIsNone(phases[-1]["end_seconds"])

    def test_nested_markers_do_not_hide_parent_stage_end(self):
        events = [{"stage": "first_update", "event": "start", "elapsed_seconds": 0},
                  {"stage": "first_forward", "event": "start", "elapsed_seconds": 1},
                  {"stage": "first_backward", "event": "start", "elapsed_seconds": 2},
                  {"stage": "first_optimizer", "event": "start", "elapsed_seconds": 3},
                  {"stage": "first_update", "event": "end", "elapsed_seconds": 4}]
        phases = timeline_phases(events)
        self.assertTrue(phases[0]["closed"])
        self.assertEqual(phases[0]["end_seconds"], 4)
        for child in phases[1:]:
            self.assertFalse(child["closed"])
            self.assertIsNone(child["end_seconds"])

    def test_end_matches_nearest_unclosed_same_stage(self):
        events = [{"stage": "outer", "event": "start", "elapsed_seconds": 0},
                  {"stage": "marker", "event": "start", "elapsed_seconds": 1},
                  {"stage": "outer", "event": "start", "elapsed_seconds": 2},
                  {"stage": "outer", "event": "end", "elapsed_seconds": 3},
                  {"stage": "outer", "event": "end", "elapsed_seconds": 4}]
        phases = timeline_phases(events)
        self.assertEqual(phases[2]["end_seconds"], 3)
        self.assertEqual(phases[0]["end_seconds"], 4)
        self.assertIsNone(phases[1]["end_seconds"])


if __name__ == "__main__":
    unittest.main()
