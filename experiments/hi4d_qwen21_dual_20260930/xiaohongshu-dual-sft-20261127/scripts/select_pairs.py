"""Lock the strict XHS SFT population and verify cross-frame person assignment."""
import argparse
from collections import Counter, defaultdict
from functools import lru_cache
import hashlib
import json
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


@lru_cache(maxsize=16000)
def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def ref(path):
    return {"path": str(Path(path).absolute()), "sha256": sha(path)}


def write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(data, stream, ensure_ascii=False, allow_nan=False)


def manifests(dataset, authors):
    merged = dataset / "pairs/pairs"
    yield "photo", "", merged / "pairs_manifest.json"
    yield "video", "author1_video_sft_", dataset / "pairs/vedios-pairs.lemon_backup_20261121/pairs_manifest.json"
    for number in (2, 3):
        [author] = list(authors.glob(f"{number}.*"))
        yield "video", f"author{number}_video_sft_", author / "two/pairs-final/sft/pairs_manifest_video.json"


def split_pairs(pairs):
    parents = {}

    def root(node):
        parents.setdefault(node, node)
        while parents[node] != node:
            parents[node] = parents[parents[node]]
            node = parents[node]
        return node

    for pair in pairs:
        group = pair["note"]
        for frame in (pair["source"], pair["target"]):
            parents[root(frame["rgb"]["sha256"])] = root(group)
    groups = defaultdict(list)
    for pair in pairs:
        groups[root(pair["note"])].append(pair)
    if len(groups) < 10:
        raise ValueError(f"Only {len(groups)} independent note/RGB groups")
    ordered = sorted(groups, key=lambda key: hashlib.sha256(f"42:{key}".encode()).digest())
    for i, group in enumerate(ordered):
        split = {0: "validation", 1: "test"}.get(i % 10, "train")
        for pair in groups[group]:
            pair.update(split=split, group=group)


def select(dataset, authors, models, output, *, device, smoke_only=False):
    import cv2
    import numpy as np
    import onnxruntime as ort
    from insightface import model_zoo
    from insightface.utils import face_align

    cv2.setNumThreads(1)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    options.inter_op_num_threads = 1
    options.log_severity_level = 3
    detector_path = models / "scrfd_10g_bnkps.onnx"
    encoder_path = models / "glintr100.onnx"
    provider = ["CPUExecutionProvider"]
    if device == "cuda":
        import torch
        torch.empty(1, device="cuda")
        provider.insert(0, "CUDAExecutionProvider")
    detector = model_zoo.get_model(str(detector_path), providers=provider, sess_options=options)
    encoder = model_zoo.get_model(str(encoder_path), providers=provider, sess_options=options)
    if any(model.session.get_providers()[0] != provider[0] for model in (detector, encoder)):
        raise RuntimeError(f"Requested identity provider unavailable: {provider[0]}")
    detector.prepare(ctx_id=-1, input_size=(640, 640))
    encoder.prepare(ctx_id=-1)
    geometry = dataset / "frames/geometry"
    geometry_index = geometry / "image_index.json"
    indexed_frames = {(row["image_id"], row["sha256"]): row for row in read(geometry_index)["images"]}

    @lru_cache(maxsize=None)
    def face_data(fid, rgb_sha, path):
        row = indexed_frames.get((fid, rgb_sha))
        if row is None:
            return None, "image_index_sha"
        files = {
            "smplx": geometry / f"smplx/sdpose_opt_pen/{fid}.pt",
            "owner": geometry / f"part/sdpose_opt_pen/owner/{fid}_owner.png",
            "part": geometry / f"part/sdpose_opt_pen/combined/{fid}_part.png",
            "keypoints": geometry / f"skeleton/sdpose_2d_kp/{fid}_keypoints.json",
            **{role: geometry / f"skeleton/sdpose_2d/{slot}/{fid}_skeleton.png"
               for role, slot in (("combined_pose", "combined"), ("p0", "p0"), ("p1", "p1"))},
        }
        if not all(file.is_file() for file in files.values()):
            return None, "missing_geometry"
        h, w = row["height"], row["width"]
        image = cv2.imread(str(path), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        if image is None or image.shape[:2] != (h, w):
            return None, "rgb_dimensions"
        keypoints = read(files["keypoints"])
        regions = []
        for slot in ("p0", "p1"):
            person = keypoints["people"].get(slot)
            if person is None:
                return None, "missing_sdpose_slot"
            xy = np.asarray(person["points134"], dtype=np.float32)[24:92]
            score = np.asarray(person["scores134"], dtype=np.float32)[24:92]
            valid = np.isfinite(xy).all(1) & (score >= 0.30)
            valid &= (xy >= 0).all(1) & (xy < [w, h]).all(1)
            if valid.sum() < 12:
                return None, "sdpose_face_support"
            regions.append(np.stack((xy[valid].min(0), xy[valid].max(0))))
        boxes, points = detector.detect(image)
        if points is None or len(boxes) < 2:
            return None, "scrfd_two_faces"
        scores = np.zeros((2, len(boxes)))
        for slot, (lo, hi) in enumerate(regions):
            for index, box in enumerate(boxes):
                a, b = box[:4].reshape(2, 2)
                if box[4] < 0.50 or min(b - a) < 18:
                    continue
                intersect = np.maximum(0, np.minimum(hi, b) - np.maximum(lo, a)).prod()
                union = (hi - lo).prod() + (b - a).prod() - intersect
                scores[slot, index] = intersect / max(union, 1e-8)
        ranking = sorted((scores[0, a] + scores[1, b], a, b)
                         for a in range(len(boxes)) for b in range(len(boxes))
                         if a != b and min(scores[0, a], scores[1, b]) >= 0.15)
        if not ranking or (len(ranking) > 1 and ranking[-1][0] - ranking[-2][0] < 0.02):
            return None, "ambiguous_face_slots"
        _, a, b = ranking[-1]
        crops = [face_align.norm_crop(image, landmark=points[i], image_size=112) for i in (a, b)]
        feat = encoder.get_feat(crops)
        feat /= np.maximum(np.linalg.norm(feat, axis=1, keepdims=True), 1e-8)
        return {"id": fid, "rgb": {"path": str(path), "sha256": rgb_sha},
                "hw": [h, w], "files": {k: ref(p) for k, p in files.items()},
                "points": points[[a, b]].tolist(), "features": feat.tolist(),
                "face_centers": ((boxes[[a, b], :2] + boxes[[a, b], 2:4]) / 2).tolist()}, None

    accepted, rejected = [], []
    seen = set()
    counts = Counter()
    inputs = []
    merged = dataset / "pairs/pairs"
    canonical = {}
    smoke = []
    smoke_path = output.with_name(output.stem + "-smoke.json")
    for kind, prefix, manifest in manifests(dataset, authors):
        inputs.append(ref(manifest))
        for row in read(manifest)["pairs"]:
            pid = prefix + row["pair_id"]
            counts[kind + "_enumerated"] += 1
            reason = None
            metrics = [row.get(k) for k in ("bg_sim", "l1_bg", "person_col_sim")]
            background = (all(type(value) in (int, float) for value in metrics)
                          and metrics[0] >= .83 and metrics[1] <= 26 and metrics[2] >= .52)
            if not background:
                reason = "strict_background"
            folder = merged / (prefix + row["dir"])
            paths = [folder / row[f"{slot}_file"] for slot in ("a", "b")]
            if reason is None and not all(p.is_file() for p in paths):
                reason = "missing_dataset_pair"
            if reason is None:
                digests = [sha(path) for path in paths]
                key = tuple(sorted(digests))
                if key in seen or digests[0] == digests[1]:
                    reason = "duplicate_pair"
            if reason is None:
                frames = []
                for slot, path, digest in zip(("a", "b"), paths, digests):
                    fid = Path(row[f"src_{slot}"]).stem if kind == "photo" else path.stem[2:]
                    frame, reason = face_data(fid, digest, canonical.setdefault((fid, digest), path))
                    if frame is None:
                        break
                    frames.append(frame)
            if reason is None:
                source, target = frames
                scores = np.asarray(target["features"]) @ np.asarray(source["features"]).T
                order = max(((0, 1), (1, 0)), key=lambda p: scores[0, p[0]] + scores[1, p[1]])
                if any(scores[i, order[i]] < .38 or scores[i, order[i]] - scores[i, order[1-i]] < .05
                       for i in range(2)):
                    reason = "identity_cosine_or_margin"
            if reason:
                rejected.append({"id": pid, "kind": kind, "reason": reason})
                counts[kind + "_" + reason] += 1
                continue
            seen.add(key)
            accepted.append({"id": pid, "kind": kind, "note": row.get("note") or row["video_id"],
                             "source": source, "target": target, "source_indices": list(order),
                             "identity_cosine": scores.tolist(),
                             "quality": {k: row[k] for k in ("bg_sim", "l1_bg", "person_col_sim")}})
            counts[kind + "_accepted"] += 1
            if len(smoke) < 2 and all(p["note"] != accepted[-1]["note"] for p in smoke):
                smoke.append(accepted[-1])
                if len(smoke) == 2:
                    write(smoke_path, {"pairs": [{**p, "split": "train", "group": p["note"]} for p in smoke]})
                    if smoke_only:
                        return
            if len(accepted) % 100 == 0:
                print(json.dumps(dict(counts), ensure_ascii=False), flush=True)
    split_pairs(accepted)
    report = {"counts": dict(counts), "splits": dict(Counter(p["split"] for p in accepted)),
              "split_kind_counts": dict(Counter(p["split"] + "_" + p["kind"] for p in accepted))}
    write(output, {"inputs": inputs, "geometry_index": ref(geometry_index),
                   "scrfd": ref(detector_path), "glintr100": ref(encoder_path),
                   "report": report, "pairs": accepted, "rejected": rejected})
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--authors", type=Path, required=True)
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--smoke-only", action="store_true")
    args = parser.parse_args()
    select(args.dataset, args.authors, args.models, args.output, device=args.device, smoke_only=args.smoke_only)
