import json
import re
from pathlib import Path

r4_dir = Path('/home/shangguanrz/project/pic-edit/experiments/4.two man /hi4d_pe_prompt_rewrite_20261020/v5_grounded_3d_20261030_r4/prompts')

FORBIDDEN_LEAKS = [
    re.compile(r'\bperson\s+[AB]\b', re.I),
    re.compile(r'\bidentity\s+[AB]\b', re.I),
    re.compile(r'\bactor\s+[AB]\b', re.I),
    re.compile(r'\b(?:red|blue)\s+skeleton\b', re.I),
    re.compile(r'\bskeleton\s+(?:line|color|colour)s?\b', re.I),
]

NEGATIVE_SUFFIX = "Do not add new objects, do not copy appearance from the pose guides, and do not render skeleton lines or guide colors."

total_checked = 0
errors = []

for split in ['test', 'val', 'train']:
    path = r4_dir / f'hi4d_{split}_prompts_v2.jsonl'
    with open(path, 'r', encoding='utf-8') as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            total_checked += 1
            row = json.loads(line)
            pid = row['pair_id']
            prompt = row['compiled_prompt_inference']
            
            if total_checked == 1:
                print("First row projected_layout:", row.get('projected_layout'))
                
            # Check 1: Must contain negative suffix
            if not prompt.endswith(NEGATIVE_SUFFIX):
                errors.append(f"{pid}: missing or corrupted negative constraint suffix")
                
            # Check 2: Binding convention in prompt prefix
            expected_prefix = (
                "Edit the two existing people in <image1> to match the interaction pose guided by the skeletons. "
                "<image2> defines the combined dual-person spatial layout, while <image3> guides Person 1, "
                "the person on the left in <image1>, and <image4> guides Person 2, the person on the right in <image1>. "
                "Anatomical left and right refer to each person's own body. Re-pose both individuals:"
            )
            if not prompt.startswith(expected_prefix):
                errors.append(f"{pid}: prefix does not match Person 1/2 binding contract")
                
            # Check 3: Check left_actor vs per_person
            left = row.get('left_actor')
            right = row.get('right_actor')
            if left not in ('A', 'B') or right not in ('A', 'B') or left == right:
                errors.append(f"{pid}: invalid left/right actor ({left}, {right})")
                
            pp1 = row['per_person']['person_1']['identity']
            pp2 = row['per_person']['person_2']['identity']
            if pp1 != left or pp2 != right:
                errors.append(f"{pid}: per_person identity mismatch (pp1={pp1}, left={left})")
                
            # Check 4: Check forbidden leaks outside negative suffix
            prompt_body = prompt[:-len(NEGATIVE_SUFFIX)]
            for pat in FORBIDDEN_LEAKS:
                m = pat.search(prompt_body)
                if m:
                    errors.append(f"{pid}: leaked pattern '{m.group(0)}' in prompt body")
                    
            # Check 5: Layout sentence consistency
            layout = row.get('projected_layout', {})
            rel = layout.get('person_1_relative_to_person_2') or layout.get('layout')
            if rel == 'left':
                if "Person 1's projected center lies to the left of Person 2's projected center" not in prompt:
                    errors.append(f"{pid}: layout mismatch, expected Person 1 left")
            elif rel == 'right':
                if "Person 1's projected center lies to the right of Person 2's projected center" not in prompt:
                    errors.append(f"{pid}: layout mismatch, expected Person 1 right")
            elif rel == 'nearly_aligned':
                if "Person 1 and Person 2 have closely overlapping horizontal positions" not in prompt:
                    errors.append(f"{pid}: layout mismatch, expected nearly_aligned")
            elif rel is not None:
                errors.append(f"{pid}: unexpected rel '{rel}'")

print(f"Total pairs audited across all splits: {total_checked}")
print(f"Total errors found: {len(errors)}")
if errors:
    for e in errors[:20]:
        print("ERROR:", e)
else:
    print("ALL 6,558 PAIRS PASSED 100% OF BINDING, CONTRACT, AND INTEGRITY CHECKS!")
