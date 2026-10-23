"""Compare stored official tensor shapes to this checkout's CHAMP adaptation."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    root = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(root))
    from third_party.champ_guidance import ChampGuidanceEncoder
    audit = json.loads((Path(__file__).resolve().parents[1] / "remote_depth_audit_v1.json").read_text())
    keys = audit["official_depth_weight"]["keys"]
    local = ChampGuidanceEncoder().state_dict()
    result = {
        "matched_tensor_keys": [k for k, v in local.items() if k in keys and list(v.shape) == keys[k]["shape"]],
        "mismatched_shapes": {k: {"local": list(v.shape), "official": keys[k]["shape"]}
                              for k, v in local.items() if k in keys and list(v.shape) != keys[k]["shape"]},
        "missing_local_keys": sorted(set(local) - set(keys)),
        "official_keys_not_in_local": sorted(set(keys) - set(local)),
        "official_active_trunk_elements": sum(v["numel"] for k, v in keys.items()
                                              if k.startswith(("conv_in.", "blocks.", "attentions.5."))),
        "official_conv_trunk_elements": sum(v["numel"] for k, v in keys.items()
                                            if k.startswith(("conv_in.", "blocks."))),
        "vendored_elements": sum(v.numel() for v in local.values()),
    }
    with args.output.open("x", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
        f.write("\n")
    print(json.dumps({"matched": len(result["matched_tensor_keys"]),
                      "shape_conflicts": len(result["mismatched_shapes"]),
                      "missing": len(result["missing_local_keys"])}))


if __name__ == "__main__":
    main()
