#!/usr/bin/env python3
"""How much of each rewritten prompt is per-sample signal rather than boilerplate?

The premise of this whole run is that a detailed prompt carries information the fixed
one-liner does not. If most of the 300 words repeat verbatim across pairs, the marginal
signal is small and 15 days of GPU buys little.  Measures sentence-level sharing:
a sentence that appears in many prompts is boilerplate, one that appears once is signal.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

SENTENCE = re.compile(r'(?<=[.;])\s+')
FILLER = re.compile(r'^(preserve|keep|do not|avoid|render|interpret|treat|use the|'
    r'add no|never|maintain|adapt|update|transfer|re-pose|on the source)', re.I)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--split', default='test')
    parser.add_argument('--form', default='compiled_prompt',
        choices=('compiled_prompt', 'compiled_prompt_inference'))
    args = parser.parse_args()

    path = args.out / 'prompts' / f'{args.split}.jsonl'
    rows = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
    if len(rows) < 4:
        raise SystemExit(f'need at least 4 prompts, found {len(rows)}; run assemble again later')

    doc_freq: Counter = Counter()
    per_prompt = []
    for row in rows:
        sentences = [s.strip() for s in SENTENCE.split(row[args.form]) if len(s.strip()) > 20]
        for sentence in set(sentences):
            doc_freq[sentence] += 1
        per_prompt.append((row['pair_id'], sentences))

    shared_heavy = [s for s, count in doc_freq.items() if count >= 0.6 * len(rows)]
    unique_once = [s for s, count in doc_freq.items() if count == 1]
    print(f'PROMPT {args.split} n={len(rows)} form={args.form}')
    print(f'  distinct sentences overall   : {len(doc_freq)}')
    print(f'  sentences in >=60% of prompts: {len(shared_heavy)}  '
          f'(their words: {sum(len(s.split()) for s in shared_heavy)})')
    print(f'  sentences unique to one pair : {len(unique_once)}')

    boiler_words = 0
    signal_words = 0
    for pair_id, sentences in per_prompt:
        for sentence in sentences:
            if doc_freq[sentence] >= 0.6 * len(rows):
                boiler_words += len(sentence.split())
            else:
                signal_words += len(sentence.split())
    total = boiler_words + signal_words
    print(f'  sentence-level split (>=60% shared counts as boilerplate): '
          f'{100 * boiler_words / total:.1f}% / {100 * signal_words / total:.1f}%  '
          f'-- saturated, see the word-frequency lines below')
    print(f'  mean words per prompt        : {total / len(rows):.0f}')

    # Sentence-level uniqueness saturates (PE paraphrases every time), so also report how
    # much of each prompt's word mass is carried by the bank's most common words.
    word_freq: Counter = Counter()
    for row in rows:
        word_freq.update(re.findall(r"[a-z]+", row[args.form].lower()))
    total_tokens = sum(word_freq.values())
    print(f'  distinct words in the bank     : {len(word_freq)}  (tokens {total_tokens})')
    for top in (50, 100, 200):
        covered = sum(count for _, count in word_freq.most_common(top))
        print(f'  tokens covered by top-{top:<3} words: {100 * covered / total_tokens:.1f}%')

    # Pairwise similarity of the whole prompt, as a sanity check on the split.
    def tokens(text: str) -> set:
        return set(re.findall(r"[a-z]+", text.lower()))

    sets = [tokens(row[args.form]) for row in rows]
    overlaps = []
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            union = len(sets[i] | sets[j])
            overlaps.append(len(sets[i] & sets[j]) / union if union else 0.0)
    overlaps.sort()
    print(f'  jaccard between two prompts  : median {overlaps[len(overlaps) // 2]:.3f}  '
          f'min {overlaps[0]:.3f}  max {overlaps[-1]:.3f}')
    print('  most repeated sentences:')
    for sentence, count in doc_freq.most_common(4):
        print(f'    x{count}/{len(rows)}  {sentence[:120]}')


if __name__ == '__main__':
    main()
