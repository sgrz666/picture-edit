"""Build native-pixel inspection crops and a read-only review index, not a training filter."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import statistics

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

root = Path(__file__).resolve().parent
snapshot = json.loads((root / 'snapshot.json').read_text(encoding='utf-8-sig'))
source = root / 'source'


def jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


legacy = {x['frame_id']: x for x in json.loads((source / 'qc/face_hand_quality_audit.json').read_text(encoding='utf-8'))}
gates = {x['seq'] + '_' + x['stem']: x for x in jsonl(source / 'qc/hand_gate/per_frame.jsonl')}
old_roi = {x['seq'] + '_' + x['stem']: x for x in jsonl(source / 'qc/detail_roi/detail_roi.jsonl')}
pairs = jsonl(source / 'CHAMP_v1/manifests/pairs/train_pairs_pristine_v2.jsonl')
out = root / 'inspection'
out.mkdir(exist_ok=False)
(out / 'native_crops').mkdir()
(out / 'sheets').mkdir()
font = ImageFont.truetype('C:/Windows/Fonts/arial.ttf', 16)
small = ImageFont.truetype('C:/Windows/Fonts/arial.ttf', 14)
sequence_notes = {
    '000000': '远景全身舞蹈；脸部有效像素较少；需逐帧看手部运动模糊。',
    '000111': '可见动物耳朵/鼻子效果；不作自然人脸身份和皮肤细节的干净 GT。',
    '000444': '需检查脸部整体偏软、眼镜反光与挥手拖影；服装/背景边缘不能代替细节质量。',
    '000666': '可见烟雾/白色覆盖；逐区域看遮挡，不能把覆盖区域当干净纹理 GT。',
}
records, crops = [], {}
for item in snapshot['frames']:
    frame_id = item['frame_id']
    path = root / 'input/train_rgb_pristine_v2' / (frame_id + '.png')
    assert hashlib.sha256(path.read_bytes()).hexdigest() == item['sha256']
    assert item['sha256'] == item['source_sha256']
    image = Image.open(path).convert('RGB')
    assert image.size == (1080, 1920)
    gray = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2GRAY)
    points = np.asarray(item['dwpose']['candidate'], dtype=np.float64)[:, :2]
    confidence = np.asarray(item['dwpose']['subset'], dtype=np.float64)
    row = {
        'frame_id': frame_id, 'sequence_id': item['sequence_id'], 'stem': item['stem'],
        'image_sha256': item['sha256'], 'original_size': list(image.size),
        'legacy_quality': legacy[frame_id], 'existing_geometry_gates': gates[frame_id],
        'sequence_review_note': sequence_notes[item['sequence_id']],
        'individual_texture_label': 'requires_manual_confirmation',
        'face_detail_gate_proposal': 'exclude_visible_face_effect' if item['sequence_id'] == '000111' else 'review',
        'whole_image_gt_gate_proposal': 'review_white_occlusion' if item['sequence_id'] == '000666' else 'review',
        'proposal_only': True, 'regions': {},
    }
    for name, sl in [('face', slice(24, 92)), ('hand_a', slice(92, 113)), ('hand_b', slice(113, 134))]:
        keypoints = points[sl][confidence[sl] >= 0.3]
        box = old_roi[frame_id].get(name + '_box_original')
        region = {'geometry_usable': gates[frame_id]['gates'][name].get('usable', False),
                  'geometry_side': gates[frame_id]['gates'][name].get('side'),
                  'native_hull_box_xxyy': box,
                  'texture_label': 'unreviewed', 'missing_is_not_blur': True}
        if box is None or len(keypoints) < (20 if name == 'face' else 10):
            region.update(crop=None, reason='no supported ROI; do not fabricate a sharpness score')
            row['regions'][name] = region
            crops[frame_id, name] = None
            continue
        # Existing ROI serialization is x_min,x_max,y_min,y_max, NOT xyxy.
        x0, x1, y0, y1 = map(int, box)
        padx, pady = max(8, round((x1 - x0) * .18)), max(8, round((y1 - y0) * .18))
        cropbox = (max(0, x0 - padx), max(0, y0 - pady), min(image.width, x1 + padx + 1), min(image.height, y1 + pady + 1))
        tile = image.crop(cropbox)
        relative = f'native_crops/{frame_id}_{name}.png'
        tile.save(out / relative)
        assert np.array_equal(np.asarray(tile), np.asarray(image)[cropbox[1]:cropbox[3], cropbox[0]:cropbox[2]])
        roi_gray = gray[cropbox[1]:cropbox[3], cropbox[0]:cropbox[2]]
        lap = cv2.Laplacian(roi_gray, cv2.CV_64F)
        mask = np.zeros(roi_gray.shape, dtype=np.uint8)
        shifted = np.rint(keypoints - np.array(cropbox[:2])).astype(np.int32)
        cv2.fillConvexPoly(mask, cv2.convexHull(shifted), 1)
        inside = cv2.erode(mask, np.ones((5, 5), dtype=np.uint8)).astype(bool)
        region.update(crop=relative, crop_box_xyxy=list(cropbox), crop_size=list(tile.size),
                      audit_lap_bbox=float(lap.var()),
                      audit_lap_inner_hull=float(lap[inside].var()) if inside.sum() >= 64 else None,
                      diagnostic_only='Native scores; no new pass threshold and no claim of a reliable blur detector.')
        row['regions'][name] = region
        crops[frame_id, name] = tile
    records.append(row)

by_id = {row['frame_id']: row for row in records}
for page, start in enumerate(range(0, len(records), 6), 1):
    batch = records[start:start + 6]
    sheet = Image.new('RGB', (990, 48 + 285 * len(batch)), '#f4f4f4')
    draw = ImageDraw.Draw(sheet)
    draw.text((10, 7), 'Original overview | FACE | HAND WINDOW A | HAND WINDOW B', fill='black', font=font)
    draw.text((10, 27), 'Display crops only; native pixels preserved in native_crops/. Scores are not labels.', fill='#444444', font=small)
    for index, row in enumerate(batch):
        top = 48 + index * 285
        draw.text((9, top), row['frame_id'], fill='black', font=font)
        picture = Image.open(root / 'input/train_rgb_pristine_v2' / (row['frame_id'] + '.png')).convert('RGB')
        picture.thumbnail((135, 245), Image.Resampling.LANCZOS)
        sheet.paste(picture, (10, top + 25))
        for col, name in enumerate(['face', 'hand_a', 'hand_b']):
            left = 155 + col * 275
            tile = crops.get((row['frame_id'], name))
            region = row['regions'][name]
            if tile is None:
                draw.text((left + 12, top + 115), 'ROI unavailable', fill='#777777', font=font)
                continue
            shown = tile.copy()
            factor = min(250 / tile.width, 235 / tile.height)
            shown = shown.resize((max(1, round(tile.width * factor)), max(1, round(tile.height * factor))),
                                 Image.Resampling.NEAREST if factor >= 1 else Image.Resampling.LANCZOS)
            sheet.paste(shown, (left + (250 - shown.width) // 2, top + 25 + (235 - shown.height) // 2))
            draw.text((left, top + 263), f'{tile.width}x{tile.height} native; geom={region["geometry_usable"]}', fill='#333333', font=small)
    sheet.save(out / f'sheets/review_{page:02d}.png')

frame_ids = set(by_id)
pair_frames = {str(p['sequence_id']) + '_' + str(p[side + '_stem']) for p in pairs for side in ['source', 'target']}
assert pair_frames == frame_ids
summary = {
    'schema': 'champ_pristine_detail_quality_audit_v1', 'observed_at': snapshot['observed_at'],
    'frame_count': len(records), 'sequence_counts': dict(Counter(row['sequence_id'] for row in records)),
    'native_sizes': [[1080, 1920]], 'unchanged_from_original_video_frames': True,
    'pristine_v2_pairs': len(pairs), 'pair_unique_frames': len(pair_frames), 'pairs_and_directory_match': True,
    'faces_with_width_94_to_106': sum(94 <= row['legacy_quality']['face']['face_width_px'] <= 106 for row in records),
    'unsupported_legacy_hand_metrics': sum(row['legacy_quality']['hands'][s]['visible_points'] < 12 for row in records for s in ['left_hand', 'right_hand']),
    'total_hand_slots': 2 * len(records),
    'face_width_min_median_max': [min(row['legacy_quality']['face']['face_width_px'] for row in records),
                                 statistics.median(row['legacy_quality']['face']['face_width_px'] for row in records),
                                 max(row['legacy_quality']['face']['face_width_px'] for row in records)],
    'training_manifest_modified': False, 'source_images_deleted_moved_or_overwritten': False,
    'annotations': 'Sequence-level visual flags and numerical diagnostics; individual texture labels remain unconfirmed.',
}
(out / 'records.json').write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding='utf-8')
(out / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps(summary, ensure_ascii=False, indent=2))
