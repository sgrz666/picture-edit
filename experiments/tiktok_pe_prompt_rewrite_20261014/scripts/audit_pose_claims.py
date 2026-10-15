#!/usr/bin/env python3
"""Audit rewritten prompts for claims that contradict the detected pose.

Only side-independent facts are checked, because "viewer's left" in the prompt maps to
the anatomical right in a normal photograph and TikTok selfie footage is often mirrored;
a side-sensitive check would produce false alarms, not findings.

Facts come from the DWPose body-18 layout used by dwpose_v1:
2 Rshoulder 3 Relbow 4 Rwrist 5 Lshoulder 6 Lelbow 7 Lwrist, 9/10/12/13 legs.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

SHOULDERS = (2, 5)
ELBOWS = (3, 6)
WRISTS = (4, 7)
ANKLES = (10, 13)
KNEES = (9, 12)

# Arm-specific on purpose: a bare "up"/"raised" also matches "button-up shirt",
# "zip-up jacket" and "chin raised", which is how an earlier version of this check
# reported 30 contradictions out of 46 pairs.
RAISED = re.compile(r'\b(?:arm|hand)s?\b[^.;]{0,70}?'
    r'\b(?:rais\w+|lift\w+|upward|above (?:the )?shoulder|shoulder height|high\b|'
    r'overhead|to (?:the )?(?:head|face|chin|eye) level|at (?:the )?top of the frame)\b', re.I)
LOWERED = re.compile(r'\b(?:arm|hand)s?\b[^.;]{0,70}?\b(?:lower\w*|down\w*|hang\w*|'
    r'at (?:the )?side[s]?|behind the back|on (?:the )?hip)\b', re.I)
DOWN = LOWERED
LEGS_APART = re.compile(r'\blegs?\b[^.]{0,60}\b(apart|spread|wide|split|wider)', re.I)
LEGS_TOGETHER = re.compile(r'\blegs?\b[^.]{0,60}\b(together|closed|side by side|aligned)', re.I)


def sentence_hits(text: str, pattern) -> list[str]:
    return [s.strip() for s in re.split(r'(?<=[.;])\s+', text) if pattern.search(s)]


def pose_facts(kps: list, scores: list) -> dict:
    def conf(index: int) -> bool:
        return scores[index] > 0.3

    raised = 0
    clearly_down = 0
    judgeable_arms = 0
    for shoulder, elbow, wrist in zip(SHOULDERS, ELBOWS, WRISTS):
        if not (conf(shoulder) and conf(elbow) and conf(wrist)):
            continue
        judgeable_arms += 1
        # y grows downward: wrist clearly above the shoulder reads as a raised arm,
        # wrist clearly below it reads as a lowered arm.
        if kps[wrist][1] < kps[shoulder][1] - 0.08:
            raised += 1
        elif kps[wrist][1] > kps[shoulder][1] + 0.10:
            clearly_down += 1
    ankles_ok = all(conf(index) for index in ANKLES)
    apart = None
    if ankles_ok and all(conf(index) for index in KNEES):
        apart = abs(kps[ANKLES[0]][0] - kps[ANKLES[1]][0]) > 0.15
    return {'judgeable_arms': judgeable_arms, 'raised_arms': raised,
        'both_arms_down_likely': judgeable_arms == 2 and raised == 0 and clearly_down == 2,
        'legs_apart': apart}


def audit(bank_path: Path, manifest_root: Path) -> tuple[list, int]:
    rows = [json.loads(line) for line in bank_path.read_text(encoding='utf-8').splitlines()
        if line.strip()]
    by_pair = {}
    for split in ('train', 'val', 'test'):
        path = manifest_root / f'{split}_pairs.jsonl'
        if path.is_file():
            for line in path.read_text(encoding='utf-8').splitlines():
                if line.strip():
                    row = json.loads(line)
                    by_pair[row['pair_id']] = row
    findings = []
    judged = 0
    for row in rows:
        manifest = by_pair.get(row['pair_id'])
        detection_path = manifest and manifest.get('target_dwpose_json')
        if not detection_path or not Path(detection_path).is_file():
            continue
        detection = json.loads(Path(detection_path).read_text(encoding='utf-8'))
        kps = detection.get('keypoints_norm') or []
        scores = [float(score) for score in (detection.get('scores') or [])]
        if len(kps) <= max(WRISTS + ANKLES + KNEES) or len(scores) <= max(WRISTS + ANKLES + KNEES):
            continue
        facts = pose_facts(kps, scores)
        if facts['judgeable_arms'] < 2 and facts['legs_apart'] is None:
            continue
        judged += 1
        text = row['compiled_prompt']
        raised_claims = sentence_hits(text, RAISED)
        down_claims = sentence_hits(text, DOWN)
        if facts['raised_arms'] == 0 and facts['both_arms_down_likely']:
            for claim in raised_claims:
                if not down_claims:
                    findings.append((row['pair_id'], 'arms_down_in_detections', claim[:150]))
        if facts['raised_arms'] == 2:
            for claim in down_claims:
                if not raised_claims:
                    findings.append((row['pair_id'], 'both_arms_raised_in_detections', claim[:150]))
        apart_claims = sentence_hits(text, LEGS_APART)
        together_claims = sentence_hits(text, LEGS_TOGETHER)
        if facts['legs_apart'] is False and apart_claims and not together_claims:
            findings.append((row['pair_id'], 'legs_together_in_detections', apart_claims[0][:150]))
        if facts['legs_apart'] is True and together_claims and not apart_claims:
            findings.append((row['pair_id'], 'legs_apart_in_detections', together_claims[0][:150]))
    return findings, judged


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--split', default='test')
    parser.add_argument('--manifest-root', type=Path,
        default=Path('/home/shangguanrz/project/pic-edit/datasets/TikTokDataset/TikTok_v3/manifests/pairs_strict_v1'))
    parser.add_argument('--max-print', type=int, default=12)
    args = parser.parse_args()

    bank = args.out / 'prompts' / f'{args.split}.jsonl'
    if not bank.is_file():
        raise SystemExit(f'missing {bank}; run `pe_rewrite_tiktok.py assemble` first')
    findings, judged = audit(bank, args.manifest_root)
    print(f'AUDIT split={args.split} bank={bank.name} judged={judged} contradictions={len(findings)}')
    for item in findings[:args.max_print]:
        print('  ', item[0], '|', item[1], '|', item[2])
    if findings:
        (args.out / 'qc' / f'pose_claim_contradictions_{args.split}.json').parent.mkdir(
            parents=True, exist_ok=True)
        (args.out / 'qc' / f'pose_claim_contradictions_{args.split}.json').write_text(
            json.dumps([{'pair_id': a, 'detection_fact': b, 'prompt_claim': c}
                for a, b, c in findings], ensure_ascii=False, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
