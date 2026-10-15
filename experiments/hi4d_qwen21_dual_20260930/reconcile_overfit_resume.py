import json
from pathlib import Path

directory = Path("experiments/qwen21_hi4d_dual/single_pair14_action_v2_overfit")
path = directory / "train_metrics.jsonl"
lines = path.read_text(encoding="utf-8").splitlines()
steps = [json.loads(line)["step"] for line in lines]
assert steps[:100] == list(range(1, 101)), steps[:100]
assert all(step > 100 for step in steps[100:]), steps[100:]
discarded = directory / "stage1_superseded_after_step100.jsonl"
assert not discarded.exists()
discarded.write_text("\n".join(lines[100:]) + "\n", encoding="utf-8")
path.write_text("\n".join(lines[:100]) + "\n", encoding="utf-8")
print("preserved", len(lines[100:]), "superseded rows; active metrics end at step 100")
