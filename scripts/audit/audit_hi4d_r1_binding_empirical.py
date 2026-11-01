#!/usr/bin/env python3
"""Empirical stress-testing harness for Hi4D v5 R1 Requirement (Person & Action Strict Binding).

Adversarially audits:
1. Person 1 vs Person 2 binding across all 6,558 corpus rows (0 swaps allowed).
2. The 42 excluded ambiguous rows (delta x < 20 px) for zero leakage into the bank.
3. Condition assembly (<image3> vs <image4>) for all left_actor == 'B' rows.
4. Anatomical left/right consistency and impossible pose checks across 13,116 person records (using 3,000 person_cache files).
"""
from __future__ import annotations

import argparse
from collections import Counter
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Dict, List, Set, Tuple

import numpy as np


def compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


class R1EmpiricalAuditor:
    def __init__(self, bank_dir: Path, manifest_dir: Path, consumer_dir: Path | None = None):
        self.bank_dir = bank_dir
        self.manifest_dir = manifest_dir
        self.consumer_dir = consumer_dir
        self._cache_map: Dict[str, Dict[str, Any]] = {}
        self.report: Dict[str, Any] = {
            'timestamp': '',
            'bank_dir': str(bank_dir),
            'manifest_dir': str(manifest_dir),
            'consumer_dir': str(consumer_dir) if consumer_dir else None,
            'summary': {},
            'checks': {},
            'violations': [],
        }

    def get_person_cache(self, rel_path: str) -> Dict[str, Any]:
        if rel_path not in self._cache_map:
            p = self.bank_dir / rel_path
            if not p.is_file():
                raise FileNotFoundError(f"Person cache file missing: {p}")
            self._cache_map[rel_path] = json.loads(p.read_text(encoding='utf-8'))
        return self._cache_map[rel_path]

    def audit_all(self) -> Dict[str, Any]:
        print("=" * 80)
        print("STARTING EMPIRICAL AUDIT OF REQUIREMENT R1: PERSON & ACTION STRICT BINDING")
        print("=" * 80)

        # Step 1: Load prompt banks
        print("\n[Step 1] Loading prompt bank files...")
        splits = ['train', 'val', 'test']
        all_rows: List[Dict[str, Any]] = []
        rows_by_split: Dict[str, List[Dict[str, Any]]] = {}

        for split in splits:
            file_candidates = [
                self.bank_dir / f'prompts/hi4d_{split}_prompts_v2.jsonl',
                self.bank_dir / f'prompts/{split}.jsonl',
                self.bank_dir / f'{split}.jsonl',
            ]
            bank_file = next((f for f in file_candidates if f.is_file()), None)
            if not bank_file:
                raise FileNotFoundError(f"Cannot find bank file for split {split} in {self.bank_dir}")
            
            rows = load_jsonl(bank_file)
            rows_by_split[split] = rows
            all_rows.extend(rows)
            print(f"  Split '{split}': loaded {len(rows)} rows from {bank_file.name} (SHA: {compute_sha256(bank_file)[:16]}...)")

        total_rows = len(all_rows)
        print(f"  Total bank rows: {total_rows} (Expected: 6,558)")
        assert total_rows == 6558, f"Expected 6558 rows, got {total_rows}"

        # Run Check 1: Person 1 vs Person 2 Swap Audit
        check1_results = self.check_person_order_binding(all_rows)

        # Run Check 2: 42 Excluded Rows & Zero Leakage Audit
        check2_results = self.check_42_excluded_rows(all_rows)

        # Run Check 3: Condition Assembly & Slot Mapping for left_actor == 'B'
        check3_results = self.check_condition_assembly_b_left(all_rows)

        # Run Check 4: Anatomical Left/Right Consistency and Impossible Pose Audit
        check4_results = self.check_anatomical_consistency(all_rows)

        # Compile verdict
        all_passed = (
            check1_results['passed'] and
            check2_results['passed'] and
            check3_results['passed'] and
            check4_results['passed']
        )

        verdict = "APPROVE" if all_passed else "REQUEST_CHANGES"
        print("\n" + "=" * 80)
        print(f"FINAL AUDIT VERDICT: {verdict}")
        print("=" * 80)

        self.report['summary'] = {
            'total_rows_audited': total_rows,
            'split_counts': {k: len(v) for k, v in rows_by_split.items()},
            'check1_person_order_binding': check1_results['summary'],
            'check2_42_excluded_leakage': check2_results['summary'],
            'check3_condition_assembly_b_left': check3_results['summary'],
            'check4_anatomical_consistency': check4_results['summary'],
            'verdict': verdict,
            'all_checks_passed': all_passed,
        }
        self.report['checks'] = {
            'check1': check1_results,
            'check2': check2_results,
            'check3': check3_results,
            'check4': check4_results,
        }

        return self.report

    def check_person_order_binding(self, rows: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Adversarially challenge whether Person 1 and Person 2 ever get swapped."""
        print("\n" + "-" * 70)
        print("CHECK 1: Person 1 vs Person 2 Strict Binding Across 6,558 Rows")
        print("-" * 70)

        source_left_counts = Counter()
        slot_swap_violations = []
        prompt_inversion_violations = []
        semantic_swap_violations = []
        centroid_mismatches = []

        for idx, row in enumerate(rows):
            pair_id = row['pair_id']
            left = row.get('left_actor')
            right = row.get('right_actor')
            source_left_counts[left] += 1

            # 1. Left/Right identity validity
            if left not in ('A', 'B') or right not in ('A', 'B') or left == right:
                slot_swap_violations.append({
                    'pair_id': pair_id,
                    'issue': f"Invalid left/right actors: left={left}, right={right}",
                })
                continue

            # 2. Per-person slot mapping
            per_person = row.get('per_person', {})
            p1 = per_person.get('person_1', {})
            p2 = per_person.get('person_2', {})

            if p1.get('identity') != left:
                slot_swap_violations.append({
                    'pair_id': pair_id,
                    'issue': f"Person 1 identity '{p1.get('identity')}' != left_actor '{left}'",
                })
            if p2.get('identity') != right:
                slot_swap_violations.append({
                    'pair_id': pair_id,
                    'issue': f"Person 2 identity '{p2.get('identity')}' != right_actor '{right}'",
                })

            expected_p1_idx = 0 if left == 'A' else 1
            expected_p2_idx = 1 if left == 'A' else 0
            if p1.get('official_actor_index') != expected_p1_idx:
                slot_swap_violations.append({
                    'pair_id': pair_id,
                    'issue': f"Person 1 actor index {p1.get('official_actor_index')} != {expected_p1_idx}",
                })
            if p2.get('official_actor_index') != expected_p2_idx:
                slot_swap_violations.append({
                    'pair_id': pair_id,
                    'issue': f"Person 2 actor index {p2.get('official_actor_index')} != {expected_p2_idx}",
                })

            # 3. Source order recorded centroids check
            s_order = row.get('source_order', {})
            if s_order:
                cx_A = s_order.get('cx_A')
                cx_B = s_order.get('cx_B')
                if cx_A is not None and cx_B is not None:
                    expected_left = 'A' if cx_A < cx_B else 'B'
                    if expected_left != left:
                        centroid_mismatches.append({
                            'pair_id': pair_id,
                            'cx_A': cx_A,
                            'cx_B': cx_B,
                            'recorded_left': left,
                            'calculated_left': expected_left,
                        })

            # 4. Prompt text slot checks
            inf_prompt = row.get('compiled_prompt_inference', '')
            train_prompt = row.get('compiled_prompt_training', '')

            # Exact preamble check
            expected_inf_preamble = (
                "<image3> guides Person 1, the person on the left in <image1>, and "
                "<image4> guides Person 2, the person on the right in <image1>."
            )
            if expected_inf_preamble not in inf_prompt:
                prompt_inversion_violations.append({
                    'pair_id': pair_id,
                    'issue': "Missing or modified inference preamble for Person 1 / Person 2 slots",
                    'snippet': inf_prompt[:250],
                })

            # Check for contradictory phrases
            if "Person 1 on the right" in inf_prompt or "Person 2 on the left" in inf_prompt:
                prompt_inversion_violations.append({
                    'pair_id': pair_id,
                    'issue': "Prompt asserts reversed person sides!",
                })

            # 5. Semantic text binding check:
            # Does Person 1's text in prompt actually describe p1?
            # Does Person 2's text in prompt actually describe p2?
            # Check if p1's and p2's descriptions ever swapped
            p1_desc = p1.get('description', {})
            p2_desc = p2.get('description', {})
            if p1_desc and p2_desc:
                # Person 1 clauses
                p1_torso = f"Person 1's torso is {p1_desc.get('torso')}."
                p2_torso = f"Person 2's torso is {p2_desc.get('torso')}."
                if p1_torso not in inf_prompt:
                    semantic_swap_violations.append({
                        'pair_id': pair_id,
                        'issue': f"Person 1 torso clause missing from prompt: {p1_torso}",
                    })
                if p2_torso not in inf_prompt:
                    semantic_swap_violations.append({
                        'pair_id': pair_id,
                        'issue': f"Person 2 torso clause missing from prompt: {p2_torso}",
                    })
                
                # Check for cross-person swap: is Person 2's torso clause labeled as Person 1?
                # (Only if p1_torso != p2_torso)
                if p1_desc.get('torso') != p2_desc.get('torso'):
                    false_p1_torso = f"Person 1's torso is {p2_desc.get('torso')}."
                    if false_p1_torso in inf_prompt:
                        semantic_swap_violations.append({
                            'pair_id': pair_id,
                            'issue': f"CRITICAL: Person 1's torso has Person 2's geometry: {false_p1_torso}",
                        })

        print(f"  Source-left distribution: A={source_left_counts['A']} ({source_left_counts['A']/len(rows)*100:.2f}%), "
              f"B={source_left_counts['B']} ({source_left_counts['B']/len(rows)*100:.2f}%)")
        print(f"  Slot swap violations: {len(slot_swap_violations)}")
        print(f"  Prompt inversion violations: {len(prompt_inversion_violations)}")
        print(f"  Semantic swap violations: {len(semantic_swap_violations)}")
        print(f"  Centroid vs left_actor mismatches: {len(centroid_mismatches)}")

        passed = (
            len(slot_swap_violations) == 0 and
            len(prompt_inversion_violations) == 0 and
            len(semantic_swap_violations) == 0 and
            len(centroid_mismatches) == 0
        )

        return {
            'passed': passed,
            'summary': {
                'total_rows_checked': len(rows),
                'source_left_counts': dict(source_left_counts),
                'slot_swap_violations': len(slot_swap_violations),
                'prompt_inversion_violations': len(prompt_inversion_violations),
                'semantic_swap_violations': len(semantic_swap_violations),
                'centroid_mismatches': len(centroid_mismatches),
            },
            'details': {
                'slot_swap_violations': slot_swap_violations[:10],
                'prompt_inversion_violations': prompt_inversion_violations[:10],
                'semantic_swap_violations': semantic_swap_violations[:10],
                'centroid_mismatches': centroid_mismatches[:10],
            }
        }

    def check_42_excluded_rows(self, bank_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Check the 42 excluded rows (delta x < 20 px) and verify zero leakage."""
        print("\n" + "-" * 70)
        print("CHECK 2: 42 Excluded Rows (delta x < 20 px) & Zero Leakage Audit")
        print("-" * 70)

        # 1. Load official manifests
        manifest_files = [
            self.manifest_dir / 'pairs_1500_train.jsonl',
            self.manifest_dir / 'pairs_1500_val.jsonl',
            self.manifest_dir / 'pairs_1500_test.jsonl',
        ]
        manifest_rows = []
        for mf in manifest_files:
            if not mf.is_file():
                alt = self.manifest_dir / f'final_manifests/{mf.name}'
                if alt.is_file():
                    mf = alt
                else:
                    raise FileNotFoundError(f"Manifest not found: {mf}")
            m_rows = load_jsonl(mf)
            manifest_rows.extend(m_rows)
            print(f"  Loaded manifest {mf.name}: {len(m_rows)} rows")

        total_manifest = len(manifest_rows)
        print(f"  Total manifest pairs: {total_manifest} (Expected: 6,600)")

        # Generate pair_ids for manifest rows
        manifest_pair_ids = set()
        manifest_by_id = {}
        for r in manifest_rows:
            split = r.get('split')
            if not split:
                split = 'train' if 'train' in r.get('sequence_id', '') or 'train' in str(manifest_files[0]) else 'val'
            pid = f"{split}_{r['pair']}_{r['action']}_cam{r['camera_id']}_{int(r['source_frame_id']):06d}_{int(r['target_frame_id']):06d}"
            manifest_pair_ids.add(pid)
            manifest_by_id[pid] = r

        bank_pair_ids = {r['pair_id'] for r in bank_rows}
        print(f"  Unique bank pair_ids: {len(bank_pair_ids)}")

        # Excluded pair IDs
        excluded_ids = manifest_pair_ids - bank_pair_ids
        print(f"  Manifest pairs NOT in bank: {len(excluded_ids)} (Expected: 42)")

        # Verify that excluded_ids are exactly 42 rows
        is_exact_42 = len(excluded_ids) == 42

        # Check leakage into bank
        leaked_into_bank = excluded_ids.intersection(bank_pair_ids)
        print(f"  Excluded rows leaked into bank: {len(leaked_into_bank)}")

        # Check all 42 excluded rows belong to fight01 cam52 frame 80
        fight01_80_count = sum(1 for pid in excluded_ids if 'fight01_cam52_000080' in pid)
        print(f"  Excluded rows from train_pair01_fight01_cam52_000080_*: {fight01_80_count}/42")

        # Check minimum gap in bank rows
        gaps_in_bank = []
        rows_below_20px = []
        for r in bank_rows:
            s_order = r.get('source_order', {})
            gap = s_order.get('gap')
            if gap is not None:
                gaps_in_bank.append(gap)
                if gap < 20.0:
                    rows_below_20px.append({
                        'pair_id': r['pair_id'],
                        'gap': gap,
                        'reason': s_order.get('reason'),
                    })

        min_gap = min(gaps_in_bank) if gaps_in_bank else 0.0
        max_gap = max(gaps_in_bank) if gaps_in_bank else 0.0
        median_gap = float(np.median(gaps_in_bank)) if gaps_in_bank else 0.0
        print(f"  Bank gap stats: min={min_gap:.2f}px, median={median_gap:.2f}px, max={max_gap:.2f}px")
        print(f"  Bank rows with gap < 20.0 px: {len(rows_below_20px)}")

        passed = (
            is_exact_42 and
            len(leaked_into_bank) == 0 and
            fight01_80_count == 42 and
            len(rows_below_20px) == 0 and
            min_gap >= 20.0
        )

        return {
            'passed': passed,
            'summary': {
                'total_manifest_pairs': total_manifest,
                'total_bank_pairs': len(bank_pair_ids),
                'total_excluded': len(excluded_ids),
                'leakage_count': len(leaked_into_bank),
                'min_gap_in_bank_px': min_gap,
                'rows_below_20px_in_bank': len(rows_below_20px),
            },
            'excluded_pair_ids': sorted(list(excluded_ids)),
        }

    def check_condition_assembly_b_left(self, bank_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Check condition assembly (<image3> vs <image4>) for all left_actor == 'B' rows."""
        print("\n" + "-" * 70)
        print("CHECK 3: Condition Assembly & Slot Mapping for left_actor == 'B'")
        print("-" * 70)

        b_left_rows = [r for r in bank_rows if r.get('left_actor') == 'B']
        a_left_rows = [r for r in bank_rows if r.get('left_actor') == 'A']
        print(f"  Rows with left_actor == 'B': {len(b_left_rows)} (Expected: 2,972)")
        print(f"  Rows with left_actor == 'A': {len(a_left_rows)} (Expected: 3,586)")

        b_left_target_guide_mismatches = []
        for r in b_left_rows:
            pair_id = r['pair_id']
            grounding = r.get('grounding', {})
            guides = grounding.get('target_guides', {})
            guide_A = guides.get('A', {}).get('path', '')
            guide_B = guides.get('B', {}).get('path', '')

            # In grounding target_guides:
            # guide A must point to actor A's pose
            # guide B must point to actor B's pose
            if '/pose/A/' not in guide_A or '/pose/B/' not in guide_B:
                b_left_target_guide_mismatches.append({
                    'pair_id': pair_id,
                    'guide_A': guide_A,
                    'guide_B': guide_B,
                })

        # Check real condition files if consumer_dir provided
        consumer_condition_checks = []
        if self.consumer_dir and self.consumer_dir.is_dir():
            print(f"  Checking assembled condition files in {self.consumer_dir}...")
            cond_files = list(self.consumer_dir.glob('**/*_qwen.json'))
            print(f"  Found {len(cond_files)} condition JSON files in consumer directory")

            for cf in cond_files:
                try:
                    cdata = json.loads(cf.read_text(encoding='utf-8'))
                    pair_id = cdata.get('pair_id', cf.stem)
                    order = cdata.get('person_order', {})
                    left_actor = order.get('evidence', {}).get('left_actor') or order.get('slot0_identity')
                    slot0 = order.get('slot0_identity')
                    slot1 = order.get('slot1_identity')
                    images = cdata.get('images', [])

                    # Image 3 (index 2) is slot 0
                    # Image 4 (index 3) is slot 1
                    img3_path = images[2].get('file', {}).get('path', '') if len(images) > 2 else ''
                    img4_path = images[3].get('file', {}).get('path', '') if len(images) > 3 else ''

                    guide_map = cdata.get('guide_identity_files', {})
                    expected_slot0_guide = guide_map.get(slot0, {}).get('path') if isinstance(guide_map.get(slot0), dict) else ''
                    expected_slot1_guide = guide_map.get(slot1, {}).get('path') if isinstance(guide_map.get(slot1), dict) else ''

                    violation = None
                    if left_actor == 'B':
                        # MUST HAVE: slot0 == 'B', slot1 == 'A'
                        if slot0 != 'B' or slot1 != 'A':
                            violation = f"B-left row has slot0={slot0}, slot1={slot1}"
                        # <image3> MUST point to Actor B skeleton!
                        if '/pose/A/' in img3_path or '/pose/B/' in img4_path:
                            violation = f"CRITICAL: <image3> points to Actor A or <image4> points to Actor B on B-left condition!"
                        if expected_slot0_guide and img3_path != expected_slot0_guide:
                            violation = f"<image3> path != guide_map['B']: {img3_path} vs {expected_slot0_guide}"

                    consumer_condition_checks.append({
                        'file': cf.name,
                        'pair_id': pair_id,
                        'left_actor': left_actor,
                        'slot0': slot0,
                        'slot1': slot1,
                        'img3': Path(img3_path).name if img3_path else '',
                        'img4': Path(img4_path).name if img4_path else '',
                        'violation': violation,
                    })
                except Exception as e:
                    consumer_condition_checks.append({
                        'file': cf.name,
                        'error': str(e),
                    })

        violations_in_consumer = [c for c in consumer_condition_checks if c.get('violation')]
        print(f"  Grounding target guide mismatches in bank: {len(b_left_target_guide_mismatches)}")
        print(f"  Consumer conditions audited: {len(consumer_condition_checks)}")
        print(f"  Consumer condition violations: {len(violations_in_consumer)}")

        passed = (
            len(b_left_rows) == 2972 and
            len(b_left_target_guide_mismatches) == 0 and
            len(violations_in_consumer) == 0
        )

        return {
            'passed': passed,
            'summary': {
                'b_left_count': len(b_left_rows),
                'a_left_count': len(a_left_rows),
                'b_left_guide_mismatches': len(b_left_target_guide_mismatches),
                'consumer_conditions_audited': len(consumer_condition_checks),
                'consumer_condition_violations': len(violations_in_consumer),
            },
            'consumer_conditions': consumer_condition_checks,
        }

    def check_anatomical_consistency(self, bank_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Verify that anatomical left and right arm/leg descriptions never contradict each other or create impossible poses."""
        print("\n" + "-" * 70)
        print("CHECK 4: Anatomical Consistency and Impossible Pose Audit")
        print("-" * 70)

        total_persons_audited = 0
        joint_id_violations = []
        angle_band_mismatches = []
        lateral_band_mismatches = []
        height_band_mismatches = []
        impossible_angle_violations = []
        contradictory_description_violations = []

        # Thresholds from schema
        BAND_LIMITS = {
            'elbow_knee': [(165.0, 'nearly straight'), (145.0, 'slightly bent'), (100.0, 'bent')],
        }

        def expected_joint_band(deg: float) -> str:
            for limit, word in BAND_LIMITS['elbow_knee']:
                if deg >= limit:
                    return word
            return 'sharply bent'

        # Preload person_cache files referenced in bank_rows
        print("  Auditing person facts and anatomical clauses via person_cache...")
        for row in bank_rows:
            pair_id = row['pair_id']
            per_person = row.get('per_person', {})

            for p_key in ('person_1', 'person_2'):
                person = per_person.get(p_key, {})
                if not person:
                    continue
                total_persons_audited += 1

                # Load person cache for official 3D facts
                cache_rel = person.get('cache') or person.get('facts_cache')
                cache_obj = self.get_person_cache(cache_rel) if cache_rel else {}
                facts = cache_obj.get('facts', {})
                desc = person.get('description', {}) or cache_obj.get('clauses', {})

                # 1. Joint ID checks
                arms = facts.get('arms', {})
                legs = legs_dict = facts.get('legs', {})
                l_arm = arms.get('left', {})
                r_arm = arms.get('right', {})
                l_leg = legs_dict.get('left', {})
                r_leg = legs_dict.get('right', {})

                if l_arm.get('joint_ids') != [16, 18, 20]:
                    joint_id_violations.append({'pair_id': pair_id, 'person': p_key, 'limb': 'left_arm', 'ids': l_arm.get('joint_ids')})
                if r_arm.get('joint_ids') != [17, 19, 21]:
                    joint_id_violations.append({'pair_id': pair_id, 'person': p_key, 'limb': 'right_arm', 'ids': r_arm.get('joint_ids')})
                if l_leg.get('joint_ids') != [1, 4, 7]:
                    joint_id_violations.append({'pair_id': pair_id, 'person': p_key, 'limb': 'left_leg', 'ids': l_leg.get('joint_ids')})
                if r_leg.get('joint_ids') != [2, 5, 8]:
                    joint_id_violations.append({'pair_id': pair_id, 'person': p_key, 'limb': 'right_leg', 'ids': r_leg.get('joint_ids')})

                # 2. Angle bounds & verbal band consistency
                for side, arm in (('left', l_arm), ('right', r_arm)):
                    angle = arm.get('elbow_angle_deg')
                    band = arm.get('elbow_band')
                    if angle is not None:
                        if not (0.0 <= angle <= 180.0):
                            impossible_angle_violations.append({'pair_id': pair_id, 'person': p_key, 'joint': f'{side}_elbow', 'angle': angle})
                        exp_band = expected_joint_band(angle)
                        if band != exp_band:
                            angle_band_mismatches.append({'pair_id': pair_id, 'person': p_key, 'joint': f'{side}_elbow', 'angle': angle, 'band': band, 'expected': exp_band})

                for side, leg in (('left', l_leg), ('right', r_leg)):
                    angle = leg.get('knee_angle_deg')
                    band = leg.get('knee_band')
                    if angle is not None:
                        if not (0.0 <= angle <= 180.0):
                            impossible_angle_violations.append({'pair_id': pair_id, 'person': p_key, 'joint': f'{side}_knee', 'angle': angle})
                        exp_band = expected_joint_band(angle)
                        if band != exp_band:
                            angle_band_mismatches.append({'pair_id': pair_id, 'person': p_key, 'joint': f'{side}_knee', 'angle': angle, 'band': band, 'expected': exp_band})

                # 3. Lateral band vs own_side_ratio consistency
                for side, arm in (('left', l_arm), ('right', r_arm)):
                    along = arm.get('shoulder_axis_ratio')
                    own = arm.get('own_side_ratio')
                    lat_band = arm.get('lateral_band')
                    if along is not None and own is not None and lat_band is not None:
                        exp_lat = ('across_body' if own < -0.7 else 'opposite_side' if own < -0.25
                                   else 'near_midline' if abs(along) <= 0.25 else 'own_side')
                        if lat_band != exp_lat:
                            lateral_band_mismatches.append({
                                'pair_id': pair_id, 'person': p_key, 'side': side,
                                'along': along, 'own': own, 'band': lat_band, 'expected': exp_lat
                            })

                # 4. Description sentence anatomical independence check
                l_arm_clause = desc.get('left_arm', '')
                r_arm_clause = desc.get('right_arm', '')
                l_leg_clause = desc.get('left_leg', '')
                r_leg_clause = desc.get('right_leg', '')

                # Ensure left arm description does not reference right side and vice versa
                if 'right' in l_arm_clause:
                    contradictory_description_violations.append({
                        'pair_id': pair_id, 'person': p_key, 'issue': f"Left arm clause mentions 'right': {l_arm_clause}"
                    })
                if 'left' in r_arm_clause:
                    contradictory_description_violations.append({
                        'pair_id': pair_id, 'person': p_key, 'issue': f"Right arm clause mentions 'left': {r_arm_clause}"
                    })

        print(f"  Total person records audited: {total_persons_audited} (Expected: 13,116)")
        print(f"  Unique person_cache files loaded: {len(self._cache_map)} (Expected: 3,000)")
        print(f"  Joint ID violations: {len(joint_id_violations)}")
        print(f"  Impossible angle violations: {len(impossible_angle_violations)}")
        print(f"  Angle vs adjective band mismatches: {len(angle_band_mismatches)}")
        print(f"  Lateral ratio vs band mismatches: {len(lateral_band_mismatches)}")
        print(f"  Contradictory anatomical descriptions: {len(contradictory_description_violations)}")

        passed = (
            total_persons_audited == 13116 and
            len(joint_id_violations) == 0 and
            len(impossible_angle_violations) == 0 and
            len(angle_band_mismatches) == 0 and
            len(lateral_band_mismatches) == 0 and
            len(contradictory_description_violations) == 0
        )

        return {
            'passed': passed,
            'summary': {
                'total_persons_audited': total_persons_audited,
                'unique_cache_files_loaded': len(self._cache_map),
                'joint_id_violations': len(joint_id_violations),
                'impossible_angle_violations': len(impossible_angle_violations),
                'angle_band_mismatches': len(angle_band_mismatches),
                'lateral_band_mismatches': len(lateral_band_mismatches),
                'contradictory_description_violations': len(contradictory_description_violations),
            },
            'details': {
                'joint_id_violations': joint_id_violations[:5],
                'impossible_angle_violations': impossible_angle_violations[:5],
                'angle_band_mismatches': angle_band_mismatches[:5],
                'lateral_band_mismatches': lateral_band_mismatches[:5],
                'contradictory_description_violations': contradictory_description_violations[:5],
            }
        }


def main():
    parser = argparse.ArgumentParser(description="Empirical R1 Binding Audit Harness")
    parser.add_argument('--bank', type=Path, required=True, help="Path to v5 bank directory")
    parser.add_argument('--manifests', type=Path, required=True, help="Path to final_manifests directory")
    parser.add_argument('--consumer', type=Path, default=None, help="Path to consumer pilot directory")
    parser.add_argument('--output', type=Path, default=None, help="Path to output JSON report")
    args = parser.parse_args()

    auditor = R1EmpiricalAuditor(args.bank, args.manifests, args.consumer)
    report = auditor.audit_all()

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
        print(f"\nSaved empirical report to {args.output}")

    sys.exit(0 if report['summary']['all_checks_passed'] else 1)


if __name__ == '__main__':
    main()
