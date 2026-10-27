"""Build an offline illustrated report from exported evidence, without inference."""
from pathlib import Path
import argparse
import base64
import hashlib
import html
import io
import json
import math
import mimetypes
import re


def esc(value):
    return html.escape(str(value))


def digest(data):
    return hashlib.sha256(data).hexdigest()


CSS = """
:root{color-scheme:light;--ink:#1c2530;--muted:#576574;--line:#d7dfe5;--accent:#205d72}
*{box-sizing:border-box}body{margin:0;background:#edf0f2;color:var(--ink);font:16px/1.65 'Microsoft YaHei','Noto Sans CJK SC',sans-serif}
main{max-width:1330px;margin:auto;background:white;padding:44px 40px}h1{font-size:34px;line-height:1.3;margin:10px 0 18px}h2{font-size:25px;line-height:1.4;margin:0 0 12px}h3{font-size:19px;margin:20px 0 8px}
p{margin:9px 0 14px}.kicker{font-size:13px;letter-spacing:.12em;color:var(--accent);font-weight:700}.subtitle,.small,figcaption{color:var(--muted);font-size:13px}.lede{font-size:19px;max-width:1060px}
nav{display:flex;gap:18px;flex-wrap:wrap;margin:22px 0}a{color:#155d7d}section{padding:34px 0;border-top:1px solid var(--line)}section:first-child{border:0;padding-top:0}.purpose{border-left:3px solid #6a909c;padding:8px 16px;background:#f5f8f9}
.grid{display:grid;grid-template-columns:repeat(var(--cols,4),minmax(0,1fr));gap:14px;margin:18px 0 10px}.grid.progress{--cols:5}.grid figure{margin:0;min-width:0}.grid img{width:100%;height:auto;aspect-ratio:9/16;object-fit:contain;background:#f3f4f5;border:1px solid #e1e6e9;cursor:zoom-in}.grid.crop img{aspect-ratio:1.4;object-fit:contain}.grid figcaption{line-height:1.45;padding:7px 1px;overflow-wrap:anywhere}.grid strong{display:block;font-size:14px;color:var(--ink)}
table{border-collapse:collapse;width:100%;font-size:14px;margin:17px 0}th,td{padding:10px;text-align:left;vertical-align:top;border-bottom:1px solid var(--line)}th{background:#eef3f5;font-weight:700}code{font:12px/1.5 Consolas,monospace;overflow-wrap:anywhere}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f5f6f7;padding:15px;font-size:12px;line-height:1.6}.note{background:#fff8e9;padding:13px 17px}.verdict{background:#eef5f2;padding:14px 18px}.group{margin-top:22px}.crop-title{font-weight:700;color:var(--accent)}
details{margin:14px 0;border:1px solid var(--line);padding:10px 15px}summary{font-weight:700;cursor:pointer}.chart{width:100%;max-height:270px}.tag{display:inline-block;background:#edf3f6;padding:2px 8px;margin-right:8px;font-size:12px}.metric{font-variant-numeric:tabular-nums}
dialog{border:0;padding:18px;background:#10151a;color:white;max-width:96vw;max-height:96vh}dialog::backdrop{background:#000c}dialog img{display:block;max-width:90vw;max-height:82vh;object-fit:contain;margin:auto}dialog button{font-size:16px;padding:7px 15px;cursor:pointer;float:right}dialog p{font-size:13px;margin:10px 0 0;max-width:90vw}footer{color:var(--muted);font-size:12px;margin-top:24px}
@media(max-width:850px){main{padding:24px 18px}h1{font-size:28px}.grid,.grid.progress{grid-template-columns:repeat(2,minmax(0,1fr))}table{font-size:12px}th,td{padding:7px}}
@page{size:A4 landscape;margin:12mm}
@media print{body{background:white;font-size:9pt;line-height:1.5}main{padding:0;max-width:none}h1{font-size:24pt}h2{font-size:17pt}h3{font-size:12pt}section{break-before:page;border:0;padding:0}.grid,.grid.progress{grid-template-columns:repeat(var(--cols,4),minmax(0,1fr));gap:8px}.grid img{max-height:112mm;aspect-ratio:auto;height:105mm}.grid.progress{gap:6px}.grid.progress img{height:58mm;aspect-ratio:auto}.grid.crop{break-inside:avoid}.grid.crop img{height:48mm;aspect-ratio:auto}.crop-title{break-after:avoid}.grid figcaption{font-size:8pt}.grid strong{font-size:9pt}figure,table,.purpose,.verdict,.note{break-inside:avoid}details{break-inside:auto}details:not([open])>*:not(summary){display:block}nav,dialog,.screen-only{display:none!important}.small,figcaption{font-size:8pt}table{font-size:8pt}th,td{padding:6px}pre{font-size:7pt}a{color:inherit;text-decoration:none}.chart{max-height:65mm}}
@media print{pre{padding:9px;margin:8px 0;line-height:1.35}h2,h3{break-after:avoid}}
"""


def main():
    from PIL import Image, ImageChops, ImageStat
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--evidence", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--review", type=Path, help="Explicit visual findings written after inspecting the actual results")
    p.add_argument("--draft", action="store_true")
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError("Use a new immutable report directory")
    m = json.loads((a.evidence / "manifest.json").read_text(encoding="utf8"))
    review = json.loads(a.review.read_text(encoding="utf8")) if a.review else {}
    if not a.draft:
        if not review.get("all_final_outputs_reviewed"):
            raise ValueError("A final illustrated report requires explicit review of the actual final images")
        for kind, exp in m["experiments"].items():
            if not exp["audit"] or not exp["audit"]["passed"] or not exp["completed"]:
                raise ValueError("A final report requires completed training and CPU ancestry audit")
            for pair in exp["pairs"]:
                if set(pair["progress"]) != {str(i) for i in range(30, 301, 30)}:
                    raise ValueError("Do not silently omit scheduled images")
                if len(pair["seeds"]) != 2 or not pair["controls"]:
                    raise ValueError("Final seed/branch controls are missing")
                finding = review.get("cases", {}).get(kind + ":" + pair["role"], {})
                expected = {pair["progress"]["300"], pair["target"], *pair["baseline"].values(),
                            *pair["seeds"].values(), *pair["controls"].values()}
                if not expected <= set(finding.get("reviewed_image_sha256", [])):
                    raise ValueError("Visual findings are not bound to the current final images")
    a.output.mkdir(parents=True)
    derived = a.output / "derived"
    derived.mkdir()
    encoded = {}
    image_files = {}
    derived_refs = []
    for key, item in m["images"].items():
        path = a.evidence / item["asset"]
        data = path.read_bytes()
        if digest(data) != key:
            raise ValueError("Exported image SHA changed")
        encoded[key] = "data:" + (mimetypes.guess_type(path.name)[0] or "image/png") + ";base64," + base64.b64encode(data).decode()
        image_files[key] = path
    figures = []

    def figure(key, label, note=""):
        if not key:
            return ""
        if key not in encoded:
            raise KeyError(key)
        figures.append({"image": key, "label": label, "note": note})
        return f'<figure><img data-image="{key}" alt="{esc(label)}" title="点击查看原始分辨率"><figcaption><strong>{esc(label)}</strong>{esc(note)}</figcaption></figure>'

    def grid(items, cols=4, cls=""):
        return f'<div class="grid {cls}" style="--cols:{cols}">' + "".join(figure(*v) for v in items) + "</div>"

    def section(title, purpose, body, sid=""):
        return f'<section id="{esc(sid)}"><h2>{esc(title)}</h2><p class="purpose">{esc(purpose)}</p>{body}</section>'

    def table(headers, rows):
        return "<table><thead><tr>" + "".join(f"<th>{esc(v)}</th>" for v in headers) + "</tr></thead><tbody>" + "".join("<tr>" + "".join(f"<td>{v}</td>" for v in row) + "</tr>" for row in rows) + "</tbody></table>"

    def derived_image(im, info):
        b = io.BytesIO()
        im.save(b, format="PNG")
        data = b.getvalue()
        key = digest(data)
        path = derived / (key + ".png")
        if not path.exists():
            path.write_bytes(data)
        encoded[key] = "data:image/png;base64," + base64.b64encode(data).decode()
        image_files[key] = path
        derived_refs.append({"sha256": key, "file": "derived/" + path.name, **info})
        return key

    def crop_image(key, box, pair, region):
        # One fixed target-canvas box for ALL variants. Never detect a nicer ROI
        # in each result or warp individual results to improve alignment.
        with Image.open(image_files[key]) as im:
            im = im.convert("RGB")
            canvas = pair["condition"]["images"][0]["canvas"]["canvas_hw"]
            h, w = canvas
            x0, y0, x1, y1 = box
            bw, bh = (x1 - x0) * w, (y1 - y0) * h
            margin = 0.3 * max(bw, bh)
            pixels = (max(0, math.floor(x0*w-margin)), max(0, math.floor(y0*h-margin)),
                      min(im.width, math.ceil(x1*w+margin)), min(im.height, math.ceil(y1*h+margin)))
            if pixels[2] <= pixels[0] or pixels[3] <= pixels[1]:
                return None
            return derived_image(im.crop(pixels), {"kind": "target_box_crop", "original_sha256": key,
                "pair_id": pair["pair_id"], "region": region, "pixel_box_xyxy": pixels,
                "same_box_across_variants": True, "retouch_or_warp": False})

    body = []
    final = all(e["completed"] for e in m["experiments"].values())
    status = "两套实验均完成300个成功optimizer update" if final else "阶段图文草稿 训练尚未全部完成"
    intro = '<div class="kicker">QWEN IMAGE 2.1 · NATIVE FULL V5 · 2026年10月26日</div><h1>TikTok单人与Hi4D双人固定样本训练结果</h1>'
    intro += f'<p class="lede">{status}。本轮检查原图画布、多图输入、分阶段训练、续训与控制分支能否实际运行；生成效果按下方原图对照分别判断。</p>'
    intro += '<p class="note">每套只训练1对图，另外1对用于同人物同场景观察。观察帧未进入训练采样或optimizer；这不构成跨人物泛化验证。两套实验从同一初始化独立开始。</p>'
    if review.get("overall"):
        intro += '<p class="verdict">' + esc(review["overall"]) + '</p>'
    intro += table(["组别", "训练做什么", "未训练帧看什么", "Qwen输入"], [
        ["E1 TikTok 00031", "Source0265 → Target0374；拟合指定单人姿态", "E2 Target0126：是否仍能响应另一姿态", "Source＋1张SMPL-X投影3D骨架"],
        ["E3 Hi4D pair12/highfive12/cam4", "Source000011 → Target000082；拟合双手击掌", "E4 Target000112：是否错误延续击掌", "Source＋联合骨架＋两张独立人物骨架"],
    ])
    intro += '<nav><a href="#setup">输入与训练设置</a><a href="#tiktok">TikTok结果</a><a href="#hi4d">Hi4D结果</a><a href="#technical">运行审计</a><a href="#prompts">完整提示词</a></nav><p class="small">HTML内嵌全部图片，可离线打开并点击放大。GT仅为监督／对照，不是Qwen输入。所有全图保持原文件；局部裁切统一使用目标条件ROI，另存来源坐标。</p>'
    body.append('<section>' + intro + '</section>')
    settings = table(["项目", "固定设置"], [
        ["画布", "TikTok 604×1080 → 608×1088；Hi4D 940×1280 → 960×1280。宽×高，native_pad，无512／768缩图。"],
        ["采样", "20步，true_cfg_scale=1，KV关闭；常规图seed42，最终补seed2026／3407。生成RNG与训练RNG隔离。"],
        ["训练", "单GPU、B=1、累积1、300个成功update；Qwen BF16且冻结，控制与optimizer状态FP32。"],
        ["Loss", "全程velocity-only；Face／Hand局部velocity、ID和Perceptual权重均为0。"],
        ["LR", "protect_inherited：Head／Strength 1e-4；新模块／Controller 5e-5；继承或未知Encoder 1e-5。S3乘0.1；clip norm1。"],
        ["提示词", "逐样本读取既有Plan A compiled_prompt_inference原文，训练与推理全文和SHA相同，未重跑Q4。"],
        ["外观边界", "ArcFace、DINO、全身Identity、Source Detail仅用第一张Source；骨架图不进入外观分支。"],
        ["几何来源", "TikTok使用同帧同拟合SMPL-X的Body21／Depth／局部几何，表情为配对driver，不做Source表情反事实重定向。Hi4D使用官方原生SMPL及可信2D Hybrid局部条件，不拟合SMPL-X。"],
        ["序列长度", "TikTok目标2584个图像token、参考5168；Hi4D目标4800、参考19200。文本token另计；Adapter自身不添加token。"],
    ])
    settings += table(["阶段", "成功update区间", "训练内容", "生成观察规则"], [
        ["S0", "1–60", "Body、Body21、Depth等有效基础几何", "只开基础Body；不把后续局部分支带来的变化全归于步数"],
        ["S1", "61–120", "Face／Hand／Detail／双人Interaction输出Head预热", "有效路线等权，Detail共享Face／Hand Head"],
        ["S2", "121–240", "有效分支Controller、必要Encoder及对应Strength", "每30步同时看训练帧与未训练观察帧"],
        ["S3", "241–300", "当前有效分支联合微调；继承共享Geometry仍冻结", "同一seed与采样配置；无自动超预算续训"],
    ])
    body.append(section("输入协议与四阶段设置", "本组说明保持不变的实验变量，以及哪些阶段的分支开关不同。", settings, "setup"))
    case_titles = {
        ("tiktok", "train"): ("E1 TikTok训练帧0374", "检验单样本拟合：Source0265的人物、服装和背景能否按指定骨架变换。"),
        ("tiktok", "observation"): ("E2 TikTok未训练帧0126", "检验同人物同场景的另一姿态响应。该目标未进入optimizer，不据此宣称跨人物泛化。"),
        ("hi4d", "train"): ("E3 Hi4D训练帧000082", "检验双人击掌拟合：稳定人物身份、双手接触区域及前后关系是否保持。"),
        ("hi4d", "observation"): ("E4 Hi4D未训练帧000112", "检验非接触观察帧：训练了击掌后，模型是否仍错误生成击掌或把两人手臂粘连。"),
    }
    role_names = {"source": "图1 Source 外观来源", "target_pose": "图2 目标3D骨架", "target_pose_joint": "图2 联合目标骨架",
                  "target_pose_a": "图3 稳定人物A骨架", "target_pose_b": "图4 稳定人物B骨架",
                  "person_0_pose": "图3 稳定人物A骨架", "person_1_pose": "图4 稳定人物B骨架"}
    role_names['combined_pose'] = '图2 联合双人目标骨架'
    off_stats = []
    for kind, exp in m["experiments"].items():
        for index, pair in enumerate(exp["pairs"]):
            title, why = case_titles[kind, pair["role"]]
            inputs = [(item["image"], "图2 联合双人目标骨架" if kind == "hi4d" and item["role"] == "target_pose" else role_names.get(item["role"], "输入 " + item["role"]), "进入Qwen的真实参考图") for item in pair["inputs"]]
            inputs.append((pair["target"], "Target RGB GT", "仅用于监督和结果对照，不进入生成外观条件"))
            inp = grid(inputs, len(inputs))
            inp += '<p class="small">人物编号由稳定ID映射决定，不按画面左右排序。完整提示词与SHA见文末。</p>'
            body.append(section(title + " 输入", why, inp, kind if index == 0 else ""))
            full = pair["progress"].get("300")
            last_update = max(map(int, pair["progress"]), default=0)
            comparison = [(pair["baseline"].get("qwen_only"), "Qwen only", "同样的多图与提示词，Adapter关闭"),
                          (pair["baseline"].get("native"), "V5初始化 0步", "共同初始化，尚未训练"),
                          (full or next(reversed(pair["progress"].values()), None), f"V5 {last_update}步", f"此帧训练{last_update}步" if pair["role"] == "train" else f"模型训练{last_update}步，此目标未参与训练"),
                          (pair["target"], "Target RGB GT", "真实目标帧")]
            note = ""
            if kind == "tiktok":
                note = ("重点检查笑容、手指和双臂边缘；也要检查Source的背景和衣服是否被破坏。局部变清晰不等于姿态已经准确。" if pair["role"] == "train" else
                        "重点检查手臂与GT的对应、面部重影以及是否向训练帧0374的姿态偏移。")
            else:
                note = ("Qwen-only在这个固定样本上已有较强基线，需放大击掌接触处和脸手判断增量。不能仅凭全图相似认定Interaction有效。" if pair["role"] == "train" else
                        "重点看双手是否分开、人物是否互换、接触训练是否污染非接触动作。")
            findings = review.get("cases", {}).get(kind + ":" + pair["role"], {})
            observed = '<p class="verdict"><strong>实际观察：</strong>' + esc(findings["observation"]) + '</p>' if findings.get("observation") else ""
            body.append(section(title + " 基座与最终结果", "C1 基座对照：比较完全相同输入下的Qwen-only、初始化V5及训练后V5。", grid(comparison) + observed + '<p class="note">' + note + '</p>'))
            progress = []
            for n, key in pair["progress"].items():
                stage = "S0 Body-only" if int(n) <= 60 else "S1 Heads" if int(n) <= 120 else "S2 Branch" if int(n) <= 240 else "S3 Joint"
                progress.append((key, n + " updates", stage + " · seed42"))
            body.append(section(title + " 每30步变化", "C2 过程观察：完整保留所有预定保存点，不按生成效果筛选。S0只开Body，跨阶段不属于单变量消融。", grid(progress, 5, "progress")))
            local = pair.get("local_summary")
            if local and full:
                boxes = local["face_boxes"][0] + [b for person in local["hand_boxes"][0] for b in person]
                valids = local["face_roi_valid"][0] + [v for person in local["hand_roi_valid"][0] for v in person]
                names = ("人物A脸部", "人物B脸部", "人物A解剖左手", "人物A解剖右手", "人物B解剖左手", "人物B解剖右手")
                zooms = ""
                for box, valid, name in zip(boxes, valids, names):
                    if not valid:
                        continue
                    variants = [(pair["baseline"]["qwen_only"], "Qwen only"), (full, "V5 300步"),
                                (pair["controls"].get("local_off:42"), "Local off"), (pair["target"], "GT")]
                    zooms += '<div class="crop-title">' + esc(title + " · " + name) + '</div>'
                    zooms += grid([(crop_image(key, box, pair, name), label, "同一目标ROI裁切，不进行对齐修图") for key, label in variants if key], 4, "crop")
                body.append(section(title + " 脸手细节", "固定使用条件缓存中的目标ROI；所有变体采用同一裁切框并外扩30%上下文。裁切仅供观察，不是额外生成或局部修复。", zooms))
            if pair["seeds"]:
                seeds = [(full, "seed42", "主观察种子")]
                seeds += [(pair["seeds"].get("native:" + str(s)), "seed" + str(s), "同一300步checkpoint，仅推理随机种子变化") for s in (2026, 3407)]
                seeds.append((pair["target"], "GT", "真实目标帧"))
                body.append(section(title + " 最终多种子", "C3 随机性观察：检查结论是否只依赖seed42。没有重新训练三套模型。", grid(seeds)))
            if pair["controls"]:
                controls = [(pair["controls"].get("native:42"), "V5 Full", "全部当前有效分支"),
                            (pair["controls"].get("local_off:42"), "Local off", "关闭Face、Hand、Source Detail；全身Identity仍保留"),
                            (pair["controls"].get("depth_off:42"), "Depth off", "Depth融合强度置0；Body21、owner与关系几何仍保留")]
                if kind == "hi4d":
                    controls.append((pair["controls"].get("interaction_off:42"), "Interaction off", "关闭独立Interaction及Raster／PartContact输入"))
                controls.append((pair["target"], "GT", "结果对照"))
                content = grid(controls, len(controls))
                content += '<p class="note">C4属于同一checkpoint的推理关闭对照，不是重新训练的消融。Local off同时关闭三类局部分支，不能由此分离Face、Hand与Detail的贡献。</p>'
                for key, label, _ in controls[:-1]:
                    if not key or label == "V5 Full":
                        continue
                    with Image.open(image_files[pair["controls"]["native:42"]]) as im1, Image.open(image_files[key]) as im2:
                        rect = tuple(pair["condition"]["images"][0]["canvas"]["valid_rect"])
                        diff = ImageChops.difference(im1.convert("RGB").crop(rect), im2.convert("RGB").crop(rect))
                        mean = sum(ImageStat.Stat(diff).mean)/3
                        off_stats.append([kind, pair["role"], label, f"{mean:.4f}"])
                body.append(section(title + " 控制路径关闭", "C4 路径敏感性：固定权重、seed42和采样配置，一次关闭一种控制组合，观察生成是否变化。", content))
    technical = '<p>每套模型均从同一个V5初始化文件开始。Hi4D未使用TikTok训练后的权重。技术审计与生成判断分别列出。</p>'
    rows = []
    for kind, exp in m["experiments"].items():
        audit = exp["audit"]
        if audit:
            for stage in ("S0", "S1", "S2", "S3"):
                s = audit["stages"][stage]
                rows.append([kind, stage, str(s["updates"]), f'{s["loss_velocity"]["mean"]:.6f}', f'{s["gradient_norm"]["mean"]:.6f}', esc(str(s["routes"]))])
    technical += table(["实验", "阶段", "成功更新", "训练velocity均值", "clip前梯度范数均值", "实际采样路线次数"], rows)
    technical += '<p class="small">训练velocity使用随机sigma且路线变化，以上均值不能作为跨阶段生成质量评分。固定噪声观察指标另外按相同route和seed比较。</p>'
    probe_rows = []
    for kind, exp in m["experiments"].items():
        for probe in (exp.get("audit") or {}).get("committed_observation_probes", []):
            for item in probe["records"]:
                if "objective" in item:
                    probe_rows.append([kind, str(probe["update"]), item["route"], f'{item["sigma"]:.2f}', f'{item["objective"]:.6f}'])
    if probe_rows:
        technical += '<details><summary>已提交固定噪声观察velocity记录</summary><p>每个route使用固定Generator seed 24042＋100×route序号；sigma固定0.2。阶段间route序号可能变化，因此只在相同route及种子语义下比较。此指标不是图片质量分数。</p>' + table(["实验", "更新点", "route", "sigma", "velocity"], probe_rows) + '</details>'
    technical += '<h3>分支关闭产生了多大像素变化</h3><p>下表是关闭结果与Full结果在原图有效矩形内的RGB绝对差均值，范围0–255。它只检查输出敏感性，数值越大不表示质量越好。</p>'
    technical += table(["实验", "帧用途", "关闭项", "RGB绝对差均值"], off_stats)
    technical += '<h3>权重和运行证据</h3>'
    availability = []
    for kind, exp in m["experiments"].items():
        reports = [(original, item) for original, item in exp["evidence"].items() if original.endswith("S2_data.json")]
        if reports:
            original, item = max(reports, key=lambda x: x[0])
            report = json.loads((a.evidence / item["file"]).read_text(encoding="utf8"))
            for pair in exp["pairs"]:
                rows_for_pair = [r for r in report["rows"] if r["pair_id"] == pair["pair_id"]]
                cells = {}
                for row in rows_for_pair:
                    cells[row["route"]] = row.get("effective_regions", {})
                availability.append([kind, pair["role"], esc(str(cells.get("face", {}).get("face", []))),
                    esc(str(cells.get("hand", {}).get("hand", []))), esc(str(cells.get("detail", {}).get("detail", [])))])
    technical += '<h3>实际可用区域</h3><p>以下为S2数据资格报告中的有效区域；几何字段存在不等于该区域进入了当前控制。Face顺序A／B；Hand顺序A左／A右／B左／B右；Detail为A脸／B脸／A左手／A右手／B左手／B右手。</p>'
    technical += table(["实验", "帧用途", "Face", "Hand", "Detail"], availability)
    for kind, exp in m["experiments"].items():
        if exp["audit"]:
            audit = exp["audit"]
            technical += f'<p><strong>{kind}</strong>：已提交1–300步无缺口／重复；S0／S1／S2／S3预算60／60／120／60；阶段间optimizer moments续接、冻结Qwen及辅助模块摘要一致。</p><pre>' + esc(json.dumps({"checkpoint": audit["checkpoint"], "initialization": audit["initialization"]}, ensure_ascii=False, indent=2)) + '</pre>'
    technical += '<p>运行中修复了checkpoint前缀审计hook的额外Autograd记录、Prepared条件的嵌套设备迁移，以及单人S3对无效Interaction Head的误验收。修复和失败日志保留；没有用常数填充零Head，也没有隐式解冻或改变图片／提示词。</p>'
    technical += '<p class="note">Hand与Source Detail中明确新初始化的部分仍标记fresh_unverified。Face候选编码器来自已有3000步文件；这不等于V5已经继承原Full的生成效果。Hi4D缺失的Jaw、Expression、Hand45、BPS与细局部Camera-Z继续无效。</p>'
    drift_refs = [item for exp in m['experiments'].values() for original,item in exp['evidence'].items() if '/source/drift_' in original]
    if drift_refs:
        drift = json.loads((a.evidence / drift_refs[-1]['file']).read_text(encoding='utf8'))
        changed = [name for name,value in drift['changed_files'].items() if value['protected_neural_file']]
        technical += '<h3>运行期间的工作区变更</h3><p class="note">远端工作树在2026年10月26日14:37加入可选的独立Source Bank接线，涉及' + esc(str(len(changed))) + '个受保护核心文件；本轮输入仍是一个Source加骨架，并未改成两张Source。当前工作树与开训快照的严格SHA一致性审计失败，因此不能宣称“当前源码就是开训源码”或已经证明逐位复现。旧快照与当前差异均已封存，没有回滚其他修改。</p><p>checkpoint提交、300步计数、参数布局及冻结摘要核验通过，与源码漂移结论分开报告。阶段记录的文件指纹不能单独证明长驻Python进程内已载入函数的字节身份，后续复现应明确选择归档版本并重新验收。</p><pre>' + esc(drift['reference_snapshot']) + '</pre>'
    body.append(section("训练与运行审计", "这一组证明实际更新、冻结和来源边界，不替代视觉评估。全部数字来自已提交谱系及原始PNG。", technical, "technical"))
    prompts = '<p>多图协议实际使用每条qwen_condition.prompt.text及其SHA。训练配置中的旧全局prompt字段仅供legacy兼容，不能用它判断本轮真正进入Qwen的文字；冻结缓存与推理核对的是下列逐样本全文。</p>'
    for kind, exp in m["experiments"].items():
        for pair in exp["pairs"]:
            prompt = pair["condition"]["prompt"]
            if digest(pair["prompt"].encode()) != prompt["sha256"] or prompt["text"] != pair["prompt"]:
                raise ValueError("Locked prompt conflict")
            prompts += '<h3>' + esc(case_titles[kind, pair["role"]][0]) + '</h3><p class="small">字段 compiled_prompt_inference · SHA256 <code>' + esc(prompt["sha256"]) + '</code></p><pre>' + esc(pair["prompt"]) + '</pre>'
    body.append(section("完整提示词与图片职责", "原文保留既有Plan A合成结果，不改写动作描述；训练与推理共用相同全文和SHA。", prompts, "prompts"))
    usage = '<p>阅读顺序：先看输入，再看C1基线，随后看C2完整过程、脸手细节、C3多种子及C4关闭对照。HTML点击图片可查看原始分辨率；PDF适合顺序阅读。SHA与原始远端路径见证据包manifest。</p>'
    usage += '<pre>ssh -o BindAddress=0.0.0.0 sg\ncd /home/shangguanrz/project/pic-edit\n/home/shangguanrz/nihaochen/envs/qwen21/bin/python scripts/audit/status_qwen21_fixedpair_v5.py --verify-latest</pre>'
    usage += '<h3>独立复现未训练观察帧</h3><p>下面是单例推理接口，自动读取锁定的逐样本提示词和多图顺序，不加载目标RGB。由于上面的并发源码漂移，执行前先明确选择归档版本并重新验收；不能在当前工作树重跑后直接称为逐位复现。每次使用新的output目录，不覆盖已有图片。原尺寸四图推理耗时较长。</p>'
    for kind, exp in m["experiments"].items():
        if exp["audit"]:
            command = ('/home/shangguanrz/nihaochen/envs/qwen21/bin/python scripts/run/run_qwen21_native_full_v5.py \\\n'
                '  --model /data/shangguanrz/pic-edit-models/Qwen-Image-2.1 \\\n'
                '  --checkpoint ' + exp["audit"]["checkpoint"]["path"] + ' \\\n'
                '  --manifest ' + exp["root"] + '/data/native_v5_v1/observation.jsonl \\\n'
                '  --data-profile ' + ('full_smplx' if kind == 'tiktok' else 'hybrid_hi4d') + ' \\\n'
                '  --steps 20 --seed 42 \\\n'
                '  --output ' + exp["root"] + '/output/manual_observation_20261026_v1')
            usage += '<h3>' + esc(kind) + '</h3><pre>' + esc(command) + '</pre>'
    usage += '<p>更多检查见 docs/Qwen21_V5固定样本训练使用说明_20261026.md。不要重新运行accept-and-train来“查看结果”，该命令会启动训练。</p>'
    usage += '<p>本轮尚未测量新的PCK、ArcFace身份分数、PSNR或SSIM，不借用旧实验数字；也未包含原Full／V4.1的同预算训练对照，不能据此宣称完整效果继承。后续应先固定一批额外未训练姿态，在相同预算下对比Qwen-only／V4.1／V5，再决定是否扩大训练。</p>'
    body.append(section("如何使用与结论边界", "结果文档用于复查这两套固定样本实验；避免把单样本或推理关闭对照扩大解释。", usage))
    js = """
const IMAGES=__IMAGES__;
for(const im of document.querySelectorAll('img[data-image]')){im.src=IMAGES[im.dataset.image];im.onclick=()=>{const d=document.querySelector('dialog');d.querySelector('img').src=im.src;d.querySelector('p').textContent=im.closest('figure').innerText+' | SHA256 '+im.dataset.image;d.showModal();};}
document.querySelector('dialog button').onclick=()=>document.querySelector('dialog').close();
document.querySelector('dialog').addEventListener('click',e=>{if(e.target.tagName==='DIALOG')e.target.close();});
window.REPORT_READY=true;
""".replace("__IMAGES__", json.dumps(encoded, separators=(",", ":")))
    document = '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>TikTok与Hi4D V5固定样本训练结果</title><style>' + CSS + '</style><main>' + "".join(body) + '<footer>只读报告 · 导出于 ' + esc(m["exported_at"]) + ' · 原始图片和SHA可追溯 · ' + str(len(figures)) + '个图片展示位</footer></main><dialog><button aria-label="关闭图片">关闭</button><img alt="原分辨率结果"><p></p></dialog><script>' + js + '</script></html>'
    destination = a.output / "V5固定样本实验结果.html"
    destination.write_text(document, encoding="utf8")
    # Printing the giant base64 document can exhaust Chromium's print process.
    # Each section below loads only its own original files, with identical CSS
    # and no resampling or image compression. The portable HTML stays intact.
    print_parts = a.output / 'print_parts'
    print_parts.mkdir()
    for number, block in enumerate(body):
        def image_source(match):
            key = match.group(1)
            return '<img src="' + esc(image_files[key].resolve().as_uri()) + '" data-image="' + key + '"'
        block = re.sub(r'<img data-image="([a-f0-9]+)"', image_source, block)
        part = '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><style>' + CSS + '</style><main>' + block + '</main></html>'
        (print_parts / f'part_{number:02d}.html').write_text(part, encoding='utf8')
    (a.output / "report_manifest.json").write_text(json.dumps({"schema": "native_v5_visual_report_v1", "draft": a.draft,
        "evidence_manifest_sha256": digest((a.evidence / "manifest.json").read_bytes()),
        "visual_review_sha256": digest(a.review.read_bytes()) if a.review else None,
        "document_sha256": digest(destination.read_bytes()), "displayed_figures": len(figures), "original_unique_images": len(m["images"]),
        "derived_images": derived_refs, "branch_off_rgb_difference": off_stats,
        "figures": figures, "model_optimizer_or_detection_executed": False}, ensure_ascii=False, indent=2), encoding="utf8")
    print(json.dumps({"html": str(destination.resolve()), "draft": a.draft, "figures": len(figures),
                      "original_unique_images": len(m["images"]), "html_bytes": destination.stat().st_size}, ensure_ascii=False))


if __name__ == "__main__":
    main()
