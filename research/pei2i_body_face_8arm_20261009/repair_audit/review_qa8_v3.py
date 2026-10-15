"""Validate downloaded PE evidence and preserve the actual visual review."""
from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from src.pose_control.qwen21_full.prompt_study.prompts import (
    compile_prompt, parse_answer, request_sha256,
)
from src.pose_control.qwen21_full.prompt_study.protocol import PROMPT_ROLES

RUN = Path(__file__).resolve().parents[1]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# Findings come from inspecting all 24 compiled paragraphs and the eight
# source/target/depth/normal/DWPose image sets; they are not model scores.
FINDINGS = {
    'train_0002794': {
        'pg': '交叉胸前手臂与 normal/depth 一致；未猜测表情；DWPose 与网格手臂不一致，PG 仍给出具体手指/手掌位置。',
        'pb': '描述胸前交叉双臂和 B 张嘴表情，保留 A 格纹衬衣、项链和室内场景；姿态描述无需读取 B RGB 执行。',
        'pbg': '描述站直、交叉双臂、B 张嘴表情；保留 A 的身份、格纹和原背景。',
    },
    'train_0000092': {
        'pg': 'viewer-right 手臂举至头顶，另一手臂放下；未臆测表情，但握拳/张指和腿间距仍较肯定。',
        'pb': '举臂侧别与 B 一致并描述张嘴；fully straight above head with elbow bent 用语不严谨；保留 A 黑衣与头巾。',
        'pbg': '举臂/下垂手臂与 B 和网格大体一致；B 张嘴；保留 A 黑衣、头巾、项链及场景。',
    },
    'train_0007286': {
        'pg': '胸前 viewer-right 手、另一臂放下；未把网格脸推断为说话；仍提到 target skeleton/geometry，属残余依赖措辞。',
        'pb': '描述 B 低头、张嘴、胸前下垂手指；保留 A 蓝红针织衣和背景。',
        'pbg': '描述 B 低头/张嘴和胸前手部；extended downward 与 relaxed curled fingers 有局部文字歧义。',
    },
    'train_0007410': {
        'pg': '已明确保留 A 球杆和白手套，没有延续 v1 删除物品的问题；网格仅近似 B 向镜头伸手，PG 把它写成胸前屈臂/松握。',
        'pb': '描述 B 向镜头伸白手套，另一手持球杆放下；保留 A 物品、服装、身份和高尔夫场景。',
        'pbg': '描述 B 向镜头指手，保留 A 球杆和白手套；混用解剖左右与 viewer-left/right，作为残余误读风险记录。',
    },
    'train_0005242': {
        'pg': '仅从网格描述一臂胸前、一臂放下，不猜微笑；实际 B 下方手靠大腿/床沿，PG 不能可靠判断接触与表情。',
        'pb': '描述 B 前倾、闭嘴微笑、胸前指手；保留 A 白缎上衣、蓝牛仔裤和卧室。',
        'pbg': '保留 A 外观并描述 B 前倾/微笑/胸前手；仍有 from the guide images 否定式措辞，未提供这些图给 Qwen RGB。',
    },
    'train_0004247': {
        'pg': '描述由侧转为正面、一臂胸前一臂放下；未猜 B smirk；把 A 头部白色配件写成耳机，物品分类不可靠。',
        'pb': '描述 B 正面和闭嘴表情，保留 A 灰衣、短裤、背景；头部白色配件称 headband。',
        'pbg': '描述 B 正面和闭嘴表情；左右使用解剖与画面混合，水印用户名识别有细微错误，不能承诺文字像素保持。',
    },
    'train_0003344': {
        'pg': '解除交叉手臂；未臆测张嘴；姿态中的具体手指和 head upright 不足以精确表达 B 低头，仍有 geometry controls 措辞。',
        'pb': '描述 B 手臂放下、低头张嘴；保留 A 眼镜、缎面衣服、房间；服装名称/肩带类型仍可能误识别。',
        'pbg': '描述 B 低头/张嘴；保留 A 眼镜和缎面纹理，但衣服被称 two-piece；仍列举编译后的几何结构指代。',
    },
    'train_0006013': {
        'pg': '保留 A 粉色卫衣和场景；网格头朝向信息弱，PG 写成朝镜头，且 viewer-right 肘弯/大腿外开与 B RGB 存在差异。',
        'pb': '描述 B 双臂外展、仰头；gaze/head 偏 viewer-right；保持 A 原图裁幅，没有要求扩出完整腿脚。',
        'pbg': '描述 B 双臂外展、仰头；gaze 偏 upper-left，与 PB 不一致，需在后续几何/人脸评估检查。',
    },
}

records = []
assets = set()
for protocol in ('pg', 'pb', 'pbg'):
    bank = RUN / f'prompts/qa8/{protocol}/train.jsonl'
    rows = [json.loads(line) for line in bank.read_text(encoding='utf-8').splitlines() if line.strip()]
    assert len(rows) == 8
    assert {row['pair_id'] for row in rows} == set(FINDINGS)
    for row in rows:
        assert row['instruction_version'] == 'instruction_v3'
        assert tuple(row['input_roles']) == PROMPT_ROLES[protocol]
        assert row['request_sha256'] == request_sha256(row['request_signature'])
        raw_path = RUN / row['raw_path']
        assert digest(raw_path) == row['raw_sha256']
        raw = json.loads(raw_path.read_text(encoding='utf-8'))
        parsed = parse_answer(raw['raw_output'])
        assert parsed['rewritten_prompt'] == row['rewritten_prompt']
        assert raw['compiled_prompt'] == row['compiled_prompt']
        assert compile_prompt(parsed['rewritten_prompt'], row['input_roles']) == row['compiled_prompt']
        assert hashlib.sha256(row['compiled_prompt'].encode()).hexdigest() == row['compiled_prompt_sha256']
        assert parsed.get('ratio_follow') == '<image1>' and parsed.get('wh_ratio') == ''
        for item in row['inputs']:
            path = RUN / item['asset_relative_path']
            assert path.resolve().is_relative_to(RUN.resolve())
            assert digest(path) == item['sha256']
            assets.add(str(path.relative_to(RUN)))
        records.append({
            'pair_id': row['pair_id'], 'protocol': protocol,
            'request_sha256': row['request_sha256'],
            'raw_sha256': row['raw_sha256'],
            'compiled_prompt_sha256': row['compiled_prompt_sha256'],
            'word_count': len(row['compiled_prompt'].split()),
            'seconds': row['seconds'],
            'review_note': FINDINGS[row['pair_id']][protocol],
        })

payload = {
    'schema': 'qwen21_qa8_v3_visual_review_v1',
    'reviewed_at': datetime.now(timezone.utc).isoformat(),
    'instruction_version': 'instruction_v3',
    'reviewed_records': len(records),
    'verified_unique_input_assets': len(assets),
    'raw_answer_parse_and_hash_checks_passed': True,
    'decision': 'release_for_controlled_experiment_with_recorded_prompt_errors',
    'notes': [
        '逐条对照了 PG/PB/PBG 各 8 条（24 条）的编译文本与全部八组 A/B/depth/normal/filtered DWPose 图片；原始回答 JSON 与最终文本、请求签名及本地图片 SHA 一致。',
        'v3 已修复删除源球杆和强制冻结 A 表情的系统性问题。PG 无目标 RGB，没有以裸网格脸宣称闭嘴/微笑/说话；PB/PBG 根据 B 可见表情描述。所有协议均明确要求 A 身份、衣服材质、物品和场景保持，允许姿态褶皱与局部阴影变化。',
        '放行用于测量自动 PE 的实际收益和错误，不代表 24 条均精确。未手改单条文本、未筛选替换样本、未读独立测试集输出，也未把 v1/v2 记录计为正式结果。',
        '残余问题：PG 的头向、细手势/腿间距可能猜测；网格与 DWPose 不一致时仍会选定一种；少数输出仍含 skeleton/geometry/guide 的措辞；PB/PBG 有手指、服装类型、水印文本误识别及左右混用，6013 的视线方向不一致。逐条备注已保留。',
        'Qwen 参考 RGB 仍仅 A，B RGB 仅在 PB/PBG 的 PE 与合法训练监督中读取；几何 adapter 单独提供数值条件。PE 文字不是保证几何正确的硬约束。',
    ],
    'records': records,
}
out = RUN / 'repair_audit/qa8_v3_visual_review.json'
out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding='utf-8')
print(json.dumps({k: v for k, v in payload.items() if k != 'records'}, ensure_ascii=False, indent=2))
