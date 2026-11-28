"""Read the official InterHand2.6M release and export immutable training rows."""

from bisect import bisect_left, bisect_right
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from functools import lru_cache
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import tarfile

import numpy as np
from PIL import Image
import torch

from src.data.canvas_contract import native_pad
from src.pose_edit.common import sha256_file

from .appearance import APPEARANCE_SIZE
from .conditions import HandCondition, JOINT_PARENTS, JOINT_PARTS, UNKNOWN
from .runtime import condition_record


SPLITS = ("train", "val", "test")
HANDS = ("left", "right")  # InterHand is one actor; Adapter slots 0/1 are A-L/A-R.
HAND_TYPES = {"left": ("left",), "right": ("right",), "interacting": HANDS}
RIGHT_ORDER = (20, 3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8, 15, 14, 13, 12, 19, 18, 17, 16)
JOINT_ORDER = np.array((np.array(RIGHT_ORDER) + 21, RIGHT_ORDER))
PROMPT = "Edit the hand pose to match the target geometry while preserving the person's appearance."


@dataclass(frozen=True)
class Camera:
    rotation: np.ndarray
    center_mm: np.ndarray
    focal: np.ndarray
    principal_point: np.ndarray

    @classmethod
    def from_annotation(cls, calibration, camera):
        keys = ("camrot", "campos", "focal", "princpt")
        shapes = ((3, 3), (3,), (2,), (2,))
        values = [np.asarray(calibration[key][camera], dtype=np.float64).reshape(shape)
                  for key, shape in zip(keys, shapes)]
        rotation, _, focal, _ = values
        if (not all(np.isfinite(value).all() for value in values) or np.any(focal <= 0)
                or not np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-3)
                or not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-3)):
            raise ValueError(f"Invalid InterHand camera calibration: {camera}")
        return cls(*values)

    def world_to_camera(self, world_m):
        return (world_m - self.center_mm / 1000) @ self.rotation.T

    def project(self, camera_m, canvas):
        depth = camera_m[..., 2]
        valid = np.isfinite(camera_m).all(axis=-1) & (depth > 0)
        safe = np.where(valid, depth, 1)
        xy = np.where(valid[..., None], camera_m[..., :2], 0) / safe[..., None]
        xy = canvas.pixel_points(xy * self.focal + self.principal_point)
        return xy, valid & np.isfinite(xy).all(axis=-1)


@dataclass(frozen=True)
class Frame:
    split: str
    capture: str
    subject: str
    sequence: str
    camera: str
    index: int
    path: str
    hw: tuple[int, int]
    hand_type: str
    view_joints: np.ndarray

    @property
    def group(self):
        return f"InterHand2.6M:{self.subject}"


def load_subject_splits(path):
    """Keep a subject in its first official split in train/val/test order."""
    captures = {}
    subject_splits = defaultdict(set)
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.partition("#")[0].split()
        if not line:
            continue
        subject, *locations, _gender = line
        if not locations:
            raise ValueError(f"Subject {subject} has no official capture")
        for location in locations:
            split, capture = location.split("/")
            if split not in SPLITS or not capture.startswith("Capture") or location in captures:
                raise ValueError(f"Invalid or duplicate InterHand capture: {location}")
            captures[location] = subject
            subject_splits[subject].add(split)
    owners = {subject: min(splits, key=SPLITS.index) for subject, splits in subject_splits.items()}
    excluded = {capture: subject for capture, subject in captures.items()
                if owners[subject] != capture.split("/")[0]}
    return owners, excluded


class _JSONStream:
    def __init__(self, handle, chunk_size):
        self.handle = handle
        self.chunk_size = chunk_size
        self.decoder = json.JSONDecoder()
        self.buffer = ""
        self.position = 0

    def fill(self):
        chunk = self.handle.read(self.chunk_size)
        self.buffer = self.buffer[self.position:] + chunk
        self.position = 0
        return bool(chunk)

    def next_char(self):
        while True:
            while self.position < len(self.buffer):
                char = self.buffer[self.position]
                if char not in " \t\r\n":
                    return char
                self.position += 1
            if not self.fill():
                raise ValueError("Unexpected end of InterHand annotation JSON")

    def consume(self, expected):
        if self.next_char() != expected:
            raise ValueError(f"Expected {expected!r} in InterHand annotation JSON")
        self.position += 1

    def value(self):
        self.next_char()
        while True:
            try:
                value, end = self.decoder.raw_decode(self.buffer, self.position)
            except json.JSONDecodeError:
                if not self.fill():
                    raise
            else:
                # A scalar at the boundary can be only the prefix of a number.
                if end == len(self.buffer) and self.fill():
                    continue
                self.position = end
                return value


def iter_coco(path, *, chunk_size=1 << 20, section=None):
    """Stream the two COCO record arrays without loading the 0.9 GB train JSON."""
    with path.open(encoding="utf-8") as handle:
        reader = _JSONStream(handle, chunk_size)
        reader.consume("{")
        while reader.next_char() != "}":
            key = reader.value()
            reader.consume(":")
            if key in ("images", "annotations"):
                reader.consume("[")
                while reader.next_char() != "]":
                    record = reader.value()
                    if section is None or key == section:
                        yield key, record
                    if reader.next_char() == "]":
                        break
                    reader.consume(",")
                reader.consume("]")
                if key == section:
                    return
            else:
                reader.value()
            if reader.next_char() == "}":
                break
            reader.consume(",")
        reader.consume("}")


class _PartReader(io.RawIOBase):
    """Seek over uncompressed tar parts while holding one part open at a time."""

    def __init__(self, parts):
        self.parts = parts
        self.offsets = [0]
        for part in parts:
            self.offsets.append(self.offsets[-1] + part.stat().st_size)
        self.position = 0
        self.part_index = -1
        self.handle = None

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=os.SEEK_SET):
        bases = {os.SEEK_SET: 0, os.SEEK_CUR: self.position, os.SEEK_END: self.offsets[-1]}
        position = bases[whence] + offset
        if position < 0:
            raise ValueError("Cannot seek before the image archive")
        self.position = position
        return position

    def readinto(self, output):
        count = 0
        while count < len(output) and self.position < self.offsets[-1]:
            index = bisect_right(self.offsets, self.position) - 1
            if index != self.part_index:
                if self.handle is not None:
                    self.handle.close()
                self.handle = self.parts[index].open("rb")
                self.part_index = index
            end = min(len(output), count + self.offsets[index + 1] - self.position)
            self.handle.seek(self.position - self.offsets[index])
            read = self.handle.readinto(memoryview(output)[count:end])
            if not read:
                raise OSError(f"Image archive part changed while reading: {self.parts[index]}")
            self.position += read
            count += read
        return count

    def close(self):
        if self.handle is not None:
            self.handle.close()
        super().close()


class TarImages:
    def __init__(self, parts, *, partial=False):
        self.parts = tuple(parts)
        self.partial = partial
        self.index = {}

    def __enter__(self):
        self.reader = io.BufferedReader(_PartReader(self.parts))
        try:
            self.archive = tarfile.open(fileobj=self.reader, mode="r:", ignore_zeros=True)
        except BaseException:
            self.reader.close()
            raise
        return self

    def __exit__(self, *_):
        self.archive.close()
        self.reader.close()

    def members(self, split=None, accept=lambda _: True):
        # tarfile normally keeps every member. Our filtered index owns the useful offsets.
        while True:
            try:
                entry = self.archive.next()
            except tarfile.ReadError:
                if not self.partial or self.archive.offset < self.reader.raw.offsets[-1]:
                    raise
                return
            self.archive.members.clear()
            if entry is None:
                return
            parts = PurePosixPath(entry.name).parts
            if len(parts) < 6 or parts[-6] != "images":
                continue
            name = "/".join(parts[-5:])
            if (entry.isfile() and parts[-5] in SPLITS and (split is None or split == parts[-5])
                    and parts[-1].lower().endswith((".jpg", ".jpeg", ".png")) and accept(name)
                    and entry.offset_data + entry.size <= self.reader.raw.offsets[-1]):
                if name in self.index:
                    raise ValueError(f"Image archive contains repeated member: {name}")
                self.index[name] = entry
                yield name

    def read(self, path):
        with self.archive.extractfile(self.index[path]) as handle:
            return handle.read()


def archive_parts(root, *, allow_partial):
    checksums = sorted(root.glob("*.tar.CHECKSUM"))
    if len(checksums) != 1:
        raise ValueError(f"Expected one official image tar CHECKSUM file in {root}")
    entries = [line.split() for line in checksums[0].read_text(encoding="utf-8").splitlines() if line.strip()]
    if (not entries or any(len(entry) != 2 or len(entry[0]) != 32 for entry in entries)
            or any(Path(entry[1]).name != entry[1] for entry in entries)):
        raise ValueError(f"Invalid image archive part list in {checksums[0]}")
    digests = {name: digest.lower() for digest, name in entries}
    names = list(digests)
    if len(names) != len(entries):
        raise ValueError(f"Repeated image archive part in {checksums[0]}")
    available = [root / name for name in names if (root / name).is_file()]
    missing = [name for name in names if not (root / name).is_file()]
    if not available or (missing and not allow_partial):
        raise ValueError(f"Image archive has {len(available)}/{len(names)} parts; use --allow-partial-archive for the available prefix")
    prefix = names[:names.index(missing[0])] if missing else names
    if not prefix:
        raise ValueError(f"First image archive part is missing: {names[0]}")
    # The split command produces equal-sized parts and a final shorter part.
    # Downloading parts can have the official filename before the bytes arrive.
    nominal = max(part.stat().st_size for part in available)
    for index, name in enumerate(prefix):
        part = root / name
        if part.stat().st_size == nominal and name != names[-1]:
            continue
        with part.open("rb") as handle:
            digest = hashlib.file_digest(handle, "md5").hexdigest()
        if digest != digests[name]:
            if not allow_partial or index == 0:
                raise ValueError(f"Incomplete or corrupt image archive part: {part}")
            return [root / name for name in prefix[:index]], missing, part.name
    return [root / name for name in prefix], missing, None


def annotation_frames(path, split, owners, excluded, exists, *, capture=None, camera=None,
                      sequence=None, hand_type=None):
    images = {}
    for _, image in iter_coco(path, section="images"):
        subject, capture_id = str(image["subject"]), str(image["capture"])
        location = f"{split}/Capture{capture_id}"
        if location in excluded and excluded[location] != subject:
            raise ValueError(f"InterHand subject metadata disagrees for {location}: {subject}")
        if subject not in owners:
            raise ValueError(f"Unknown InterHand subject {subject} in {location}")
        if (owners[subject] != split or (capture is not None and capture != capture_id)
                or (camera is not None and camera != str(image["camera"]))
                or (sequence is not None and sequence != image["seq_name"])):
            continue
        relative = PurePosixPath(image["file_name"])
        expected = (f"Capture{capture_id}", image["seq_name"], f"cam{image['camera']}")
        frame_digits = relative.stem.removeprefix("image")
        if (relative.parts[:-1] != expected or ".." in relative.parts
                or not relative.stem.startswith("image") or not frame_digits.isascii() or not frame_digits.isdecimal()
                or int(frame_digits) != int(image["frame_idx"])):
            raise ValueError(f"InterHand image path disagrees with its annotation: {relative}")
        name = f"{split}/{relative}"
        if exists(name):
            images[image["id"]] = (image, name)
    if not images:
        return
    for section, annotation in iter_coco(path):
        if section != "annotations" or annotation["image_id"] not in images:
            continue
        image, name = images.pop(annotation["image_id"])
        kind = annotation["hand_type"]
        if kind not in HAND_TYPES:
            raise ValueError(f"Unknown InterHand hand type: {kind}")
        if annotation["hand_type_valid"] and (hand_type is None or kind == hand_type):
            mask = np.asarray(annotation["joint_valid"]).reshape(42) == 1
            yield Frame(split, str(image["capture"]), str(image["subject"]), image["seq_name"],
                        str(image["camera"]), int(image["frame_idx"]), name,
                        (int(image["height"]), int(image["width"])), kind, mask)
        if not images:
            return


def select_pairs(frames, *, min_gap, max_gap, accepts=lambda _a, _b: True, targets_per_source=1):
    if targets_per_source < 1:
        raise ValueError("targets_per_source must be positive")
    groups = defaultdict(list)
    for frame in frames:
        key = (frame.split, frame.subject, frame.capture, frame.sequence, frame.camera, frame.hand_type, frame.hw)
        groups[key].append(frame)
    for key in sorted(groups):
        sequence = sorted(groups[key], key=lambda frame: (frame.index, frame.path))
        indexes = [frame.index for frame in sequence]
        for offset, source in enumerate(sequence):
            lo = bisect_left(indexes, source.index + min_gap, lo=offset + 1)
            hi = bisect_right(indexes, source.index + max_gap, lo=lo)
            remaining = targets_per_source
            for target in sequence[lo:hi]:
                if source.index != target.index and accepts(source, target):
                    yield source, target
                    remaining -= 1
                    if remaining == 0:
                        break


def frame_condition(frame, camera, world_mm, world_valid, canvas):
    world = np.asarray(world_mm, dtype=np.float64).reshape(42, 3)[JOINT_ORDER]
    world_valid = np.asarray(world_valid).reshape(42)[JOINT_ORDER] == 1
    present = np.isin(HANDS, HAND_TYPES[frame.hand_type])
    valid_3d = world_valid & world_valid[:, :1] & np.isfinite(world).all(axis=-1) & present[:, None]
    camera_m = camera.world_to_camera(world / 1000)
    xy, positive_depth = camera.project(camera_m, canvas)
    valid_2d = valid_3d & positive_depth
    joints = np.zeros((4, 21, 3), dtype=np.float32)
    joints[:2, :, :2] = np.where(valid_2d[..., None], xy, 0)
    joints[:2, :, 2] = valid_2d
    joints_3d = np.zeros_like(joints)
    joints_3d[:2] = np.where(valid_3d[..., None], camera_m, 0)
    empty = torch.zeros((4, 21), dtype=torch.bool)
    return HandCondition(
        canvas=canvas, person_ids=(frame.group, None), joints=torch.from_numpy(joints), visible=empty,
        joints_3d=torch.from_numpy(joints_3d),
        joint_3d_valid=torch.from_numpy(np.pad(valid_3d, ((0, 2), (0, 0)))),
        joint_2d_valid=torch.from_numpy(np.pad(valid_2d, ((0, 2), (0, 0)))),
        joint_occluded=empty.clone(), hand_present=torch.from_numpy(np.pad(present, (0, 2))),
    ).validate()


def _json(path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _image_bytes(source, relative):
    if isinstance(source, TarImages):
        return source.read(relative)
    return (source / relative).read_bytes()


def _verify_image(data, frame):
    with Image.open(io.BytesIO(data)) as image:
        if image.size != frame.hw[::-1]:
            raise ValueError(f"Image dimensions disagree with InterHand annotation: {frame.path}")
        image.load()


def _source_part_distances(xy, valid, size):
    """Projected source bone/point distances, in the source crop pixel grid."""
    distances = np.full((16, size, size), np.inf, dtype=np.float32)
    y, x = np.ogrid[:size, :size]
    x, y = x.astype(np.float32), y.astype(np.float32)
    for joint in np.flatnonzero(valid):
        parent = JOINT_PARENTS[joint]
        point = xy[joint]
        start = xy[parent] if valid[parent] else point
        dx, dy = point - start
        offset_x, offset_y = x - start[0], y - start[1]
        length = dx * dx + dy * dy
        progress = np.clip((offset_x * dx + offset_y * dy) / length, 0, 1) if length > 1e-6 else 0
        distance = (offset_x - dx * progress) ** 2 + (offset_y - dy * progress) ** 2
        part = JOINT_PARTS[joint]
        np.minimum(distances[part], distance, out=distances[part])
    return distances


def _source_semantics(xy, projected, slot, size):
    valid = projected[slot]
    distances = _source_part_distances(xy[slot], valid, size)
    radius = np.clip(np.ptp(xy[slot, valid], axis=0).max() * .055, 2., size * .08)
    parts = distances.argmin(0).astype(np.int64)
    closest = distances.min(0)
    second = np.partition(distances, 1, axis=0)[1]
    foreground = closest <= radius ** 2
    certain = foreground & (second > closest + (.35 * radius) ** 2)
    other = 1 - slot
    if projected[other].any():
        peer = _source_part_distances(xy[other], projected[other], size).min(0)
        # Remove ambiguous cross-hand pixels from semantic routing. The foreground
        # still supports pooling color where a source hand is expected in the crop.
        certain &= peer > radius ** 2
    return np.where(certain, parts, UNKNOWN), foreground


def source_appearance(condition, frame, source_data, source_sha, *, size=APPEARANCE_SIZE):
    """Crop real source RGB at the source projection and bind it to its actual digest."""
    crops = torch.zeros(4, 3, size, size)
    eligible = torch.zeros(4, dtype=torch.bool)
    parts = torch.full((4, size, size), UNKNOWN, dtype=torch.long)
    masks = torch.zeros(4, size, size, dtype=torch.bool)
    identity = [None] * 4
    with Image.open(io.BytesIO(source_data)) as image:
        image = image.convert("RGB")
        h, w = frame.hw
        projected = condition.projected_joints()[:2].numpy() & frame.view_joints[JOINT_ORDER]
        source_xy = (condition.joints[:2, :, :2].numpy() - condition.canvas.offset_xy) / condition.canvas.scale_xy
        for slot, valid in enumerate(projected):
            if valid.sum() < 4:
                continue
            xy = source_xy[slot, valid]
            low, high = xy.min(0), xy.max(0)
            side = min(int(np.ceil((high - low).max() * 1.5)), h, w)
            if side < 8:
                continue
            origin = np.clip(np.floor((low + high - side) / 2), (0, 0), (w - side, h - side)).astype(int)
            left, top = origin.tolist()
            crop = image.crop((left, top, left + side, top + side)).resize((size, size), Image.Resampling.BILINEAR)
            crop_xy = ((source_xy - origin + .5) * (size / side) - .5).astype(np.float32)
            labels, mask = _source_semantics(crop_xy, projected, slot, size)
            if not mask.any():
                continue
            crops[slot] = torch.from_numpy(np.array(crop)).permute(2, 0, 1).float() / 255
            parts[slot] = torch.from_numpy(labels)
            masks[slot] = torch.from_numpy(mask)
            eligible[slot] = True
            identity[slot] = source_sha
    return dict(appearance=crops, appearance_valid=eligible, appearance_sources=tuple(identity),
                appearance_parts=parts, appearance_mask=masks)


def _asset(path, out, digest=None):
    try:
        relative = Path(os.path.relpath(path, out)).as_posix()
    except ValueError:  # Windows source and output can be on different drives.
        relative = path.as_posix()
    return {"path": relative, "sha256": digest or sha256_file(path)}


def convert_split(annotation_root, image_source, out, split, owners, excluded, *,
                  min_gap, max_gap, min_motion, limit, filters, mano=None, used_splits=None,
                  targets_per_source=1):
    paths = {kind: annotation_root / split / f"InterHand2.6M_{split}_{kind}.json"
             for kind in ("data", "camera", "joint_3d", "MANO_NeuralAnnot")}
    exists = (image_source.index.__contains__ if isinstance(image_source, TarImages)
              else lambda name: (image_source / name).is_file())
    frames = list(annotation_frames(paths["data"], split, owners, excluded, exists, **filters))
    if not frames:
        return [], {"annotated_frames": 0}
    cameras = _json(paths["camera"])
    joints = _json(paths["joint_3d"])
    mano_annotations = _json(paths["MANO_NeuralAnnot"]) if mano is not None else {}
    counters = Counter(annotated_frames=len(frames))
    used_splits = used_splits if used_splits is not None else {}
    image_assets = {}

    @lru_cache(maxsize=128)
    def calibration(capture, camera):
        return Camera.from_annotation(cameras[capture], camera)

    @lru_cache(maxsize=64)
    def project(capture, camera, index, hw, hand_type, subject):
        rec = joints[capture][str(index)]
        frame = Frame(split, capture, subject, "", camera, index, "", hw, hand_type, np.zeros(42, dtype=bool))
        return frame_condition(frame, calibration(capture, camera), rec["world_coord"],
                               rec["joint_valid"], native_pad(hw))

    def projected(frame):
        return project(frame.capture, frame.camera, frame.index, frame.hw, frame.hand_type, frame.subject)

    def accepts(source, target):
        if str(source.index) not in joints.get(source.capture, {}) or str(target.index) not in joints.get(target.capture, {}):
            counters["missing_joint_annotations"] += 1
            return False
        first, second = projected(source), projected(target)
        x0, y0, x1, y1 = second.canvas.valid_rect
        xy_first, xy_second = first.joints[..., :2], second.joints[..., :2]
        lower, upper = torch.tensor((x0, y0)), torch.tensor((x1, y1))
        in_view = ((xy_first >= lower) & (xy_first < upper) & (xy_second >= lower) & (xy_second < upper)).all(-1)
        common = first.joint_2d_valid & second.joint_2d_valid & in_view
        if (common.sum(1)[second.hand_present] < 4).any():
            counters["insufficient_projected_joints"] += 1
            return False
        motion = (xy_second[common] - xy_first[common]).square().sum(-1).mean().sqrt()
        if motion < min_motion:
            counters["insufficient_motion"] += 1
            return False
        return True

    rows = []
    for source, target in select_pairs(frames, min_gap=min_gap, max_gap=max_gap, accepts=accepts,
                                       targets_per_source=targets_per_source):
        hashes = []
        for frame in (source, target):
            if frame.path in image_assets:
                hashes.append((image_assets[frame.path]["sha256"], None))
            else:
                data = _image_bytes(image_source, frame.path)
                _verify_image(data, frame)
                hashes.append((hashlib.sha256(data).hexdigest(), data))
        if hashes[0][0] == hashes[1][0]:
            counters["same_image_sha"] += 1
            continue
        if any(used_splits.get(digest, split) != split for digest, _ in hashes):
            counters["cross_split_image_sha"] += 1
            continue
        condition = projected(target)
        source_sha, source_data = hashes[0]
        if source_data is None:
            source_data = _image_bytes(image_source, source.path)
            if hashlib.sha256(source_data).hexdigest() != source_sha:
                raise ValueError(f"Source RGB changed during export: {source.path}")
        appearance = source_appearance(projected(source), source, source_data, source_sha)
        condition = replace(condition, **appearance)
        counters["source_appearance_hands"] += int(condition.appearance_valid.sum())
        fit = {}
        if mano is not None:
            params = mano_annotations.get(target.capture, {}).get(str(target.index), {})
            condition, fit = mano.enrich(condition, target, calibration(target.capture, target.camera), params)
            counters["verified_mano_hands"] += int(condition.mano_valid.sum())
            counters["mesh_occluded_joints"] += int(condition.joint_occluded.sum())
            counters["unknown_surface_candidates"] += sum(len(pair.surface_indices) for pair in condition.pairs
                                                           if pair.surface_indices is not None)
        pair_id = hashlib.sha256(f"{source.path}\0{target.path}".encode()).hexdigest()[:24]
        pack = out / "conditions" / f"{pair_id}.pt"
        pack.parent.mkdir(parents=True, exist_ok=True)
        with pack.open("xb") as handle:
            torch.save(condition_record(condition), handle)
        for frame, (digest, data) in zip((source, target), hashes):
            used_splits[digest] = split
            if frame.path in image_assets:
                continue
            if isinstance(image_source, TarImages):
                location = out / "images" / frame.path
                location.parent.mkdir(parents=True, exist_ok=True)
                with location.open("xb") as handle:
                    handle.write(data)
            else:
                location = image_source / frame.path
            image_assets[frame.path] = _asset(location, out, digest)
        rows.append({
            "id": pair_id, "group": target.group, "split": split, "mode": "global", "prompt": PROMPT,
            "source_refs": [image_assets[source.path]], "target": image_assets[target.path],
            "condition": _asset(pack, out),
            "interhand": {"capture": target.capture, "sequence": target.sequence, "camera": target.camera,
                          "source_frame": source.index, "target_frame": target.index, "hand_type": target.hand_type,
                          "target_view_joint_valid": target.view_joints[JOINT_ORDER].tolist(),
                          "mano_fit": fit},
        })
        if limit is not None and len(rows) >= limit:
            break
    counters["emitted_pairs"] = len(rows)
    return rows, dict(counters)
