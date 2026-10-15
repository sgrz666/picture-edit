#!/usr/bin/env python3
"""Read the rewritten-prompt bank the way the trainer needs it: pair_id -> compiled prompt.

Mirrors src.pose_control.qwen21_full.prompt_study.prompts.PromptBank's surface
(``prompt_for``) so a training entry can swap in this bank without touching the cache
layer -- ``Qwen21IPERCache.pair(..., prompts=[...])`` already accepts one prompt per
sample (see src/pose_control/qwen21/runtime.py:97-109).

Usage:
    python prompt_bank.py --split test --check-against-manifest
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

EXP_ROOT = Path(__file__).resolve().parents[1]


def _find_repo_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / 'src' / 'pose_edit' / 'prompt_contract.py').is_file():
            return parent
    raise RuntimeError('run this from inside the repository')


REPO_ROOT = _find_repo_root()
PAIRS_ROOT = REPO_ROOT / 'datasets/TikTokDataset/TikTok_v3/manifests/pairs_strict_v1'
SKELETON_WORD = 'the target joint arrangement described in this instruction'


class TiktokPromptBank:
    """One compiled prompt per pair, validated on load.

    ``training`` is the word-form prompt (the VLM only receives A); ``inference`` is the
    form for the workflow generator, which receives A plus the skeleton as its second
    reference. Ask for the one that matches the consumer.
    """

    def __init__(self, path: Path | str, form: str = 'training'):
        if form not in ('training', 'inference'):
            raise ValueError(f'unknown prompt form: {form}')
        self.form = form
        self.path = Path(path)
        self.records: dict[str, str] = {}
        # TikTokV3PoseDataset.__getitem__ returns appearance/source_stem/target_stem but not
        # pair_id, so a DataLoader batch can only be joined on that triple.
        self.by_sample_key: dict[str, str] = {}
        for line in self.path.read_text(encoding='utf-8').splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            pair_id = row['pair_id']
            if pair_id in self.records:
                raise ValueError(f'duplicate pair_id in bank: {pair_id}')
            if row.get('status') != 'complete':
                raise ValueError(f'bank line is not complete: {pair_id}')
            prompt = row['compiled_prompt' if form == 'training' else 'compiled_prompt_inference']
            if not isinstance(prompt, str) or not prompt.strip():
                raise ValueError(f'empty prompt: {pair_id}')
            expected = row['compiled_prompt_sha256' if form == 'training'
                else 'compiled_prompt_inference_sha256']
            if hashlib.sha256(prompt.encode()).hexdigest() != expected:
                raise ValueError(f'prompt hash mismatch: {pair_id}')
            if form == 'training':
                if '<image' in prompt:
                    raise ValueError(f'training prompt still carries an image reference: {pair_id}')
                if SKELETON_WORD not in prompt:
                    raise ValueError(f'training prompt lost the skeleton role word: {pair_id}')
            elif '<image3>' in prompt:
                raise ValueError(f'inference prompt still carries the PE numbering: {pair_id}')
            self.records[pair_id] = prompt
            sample_key = self.sample_key(row['sequence_id'], row['source_stem'], row['target_stem'])
            if sample_key in self.by_sample_key:
                raise ValueError(f'duplicate sample key in bank: {sample_key}')
            self.by_sample_key[sample_key] = prompt
        self.source_sha256 = hashlib.sha256(self.path.read_bytes()).hexdigest()

    @staticmethod
    def sample_key(appearance: str, source_stem: str, target_stem: str) -> str:
        return f'{appearance}|{source_stem}|{target_stem}'

    def prompt_for_sample(self, sample: dict) -> str:
        """Look up by the fields a dataset sample actually carries."""
        key = self.sample_key(str(sample['appearance']), str(sample['source_stem']),
            str(sample['target_stem']))
        try:
            return self.by_sample_key[key]
        except KeyError:
            raise KeyError(f'no {self.form} prompt for sample {key}') from None

    def prompt_for(self, pair_id: str) -> str:
        try:
            return self.records[pair_id]
        except KeyError:
            raise KeyError(f'no {self.form} prompt recorded for pair {pair_id}') from None

    def prompts_for(self, pair_ids: list[str]) -> list[str]:
        return [self.prompt_for(pair_id) for pair_id in pair_ids]

    def missing(self, pair_ids) -> list[str]:
        return [pair_id for pair_id in pair_ids if pair_id not in self.records]

    def coverage(self, pair_ids) -> tuple[int, int]:
        pair_ids = list(pair_ids)
        return len(pair_ids) - len(self.missing(pair_ids)), len(pair_ids)


def manifest_pair_ids(split: str, pairs_root: Path) -> list[str]:
    rows = (pairs_root / f'{split}_pairs.jsonl').read_text(encoding='utf-8').splitlines()
    return [json.loads(line)['pair_id'] for line in rows if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--split', choices=('train', 'val', 'test'), required=True)
    parser.add_argument('--bank', type=Path, default=EXP_ROOT / 'prompts')
    parser.add_argument('--pairs-root', type=Path, default=PAIRS_ROOT)
    parser.add_argument('--form', choices=('training', 'inference'), default='training')
    parser.add_argument('--check-against-manifest', action='store_true')
    args = parser.parse_args()

    bank = TiktokPromptBank(args.bank / f'{args.split}.jsonl', form=args.form)
    print(f'BANK {bank.path} form={args.form} prompts={len(bank.records)} '
          f'source_sha256={bank.source_sha256[:12]}')
    if args.check_against_manifest:
        wanted = manifest_pair_ids(args.split, args.pairs_root)
        have, total = bank.coverage(wanted)
        print(f'COVERAGE split={args.split} {have}/{total} ({100 * have / total:.1f}%)')
        missing = bank.missing(wanted)
        if missing:
            print(f'FIRST_MISSING {missing[:5]}')


if __name__ == '__main__':
    main()
