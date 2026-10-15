#!/usr/bin/env python3
"""Generate the six editable V6.4 architecture diagrams requested for the project.

The diagrams intentionally describe the current implementation on the remote main
branch plus the current iPER detail-training working tree. They do not promote
planned ArcFace/DINO integrations to implemented components.
"""

from __future__ import annotations

import html
from pathlib import Path
from xml.etree import ElementTree as ET


OUT = Path(__file__).resolve().parent
FONT = "Microsoft YaHei"

COLORS = {
    "ink": "#18324A",
    "muted": "#60758A",
    "line": "#45657F",
    "input": "#EAF3FF",
    "input_s": "#4D86C6",
    "encode": "#E4F7F1",
    "encode_s": "#2E8B73",
    "reason": "#F1EAFB",
    "reason_s": "#7651A8",
    "adapter": "#FFF1DF",
    "adapter_s": "#D07A24",
    "detail": "#FFE9EC",
    "detail_s": "#C94C65",
    "deepgen": "#EAF6E7",
    "deepgen_s": "#4D8A42",
    "frozen": "#EFF2F5",
    "frozen_s": "#768696",
    "warn": "#FFF6D9",
    "warn_s": "#B88A19",
    "white": "#FFFFFF",
    "bg": "#F8FAFC",
}


def esc(value: object) -> str:
    # Keep draw.io's HTML line breaks as markup. ElementTree will XML-escape
    # them once, and draw.io/check_layout will decode them back to <br>.
    return html.escape(str(value), quote=True).replace("\n", "<br>")


class Diagram:
    def __init__(self, title: str, subtitle: str, width: int = 1900, height: int = 1180):
        self.title = title
        self.subtitle = subtitle
        self.width = width
        self.height = height
        self.cells: list[ET.Element] = []
        self.geometry: dict[str, tuple[float, float, float, float]] = {}
        self._id = 2

    def _new_id(self, prefix: str = "c") -> str:
        value = f"{prefix}{self._id}"
        self._id += 1
        return value

    def vertex(self, value: str, x: float, y: float, w: float, h: float, style: str, cid: str | None = None) -> str:
        cid = cid or self._new_id()
        cell = ET.Element("mxCell", {"id": cid, "value": value, "style": style, "vertex": "1", "parent": "1"})
        ET.SubElement(cell, "mxGeometry", {"x": str(x), "y": str(y), "width": str(w), "height": str(h), "as": "geometry"})
        self.cells.append(cell)
        self.geometry[cid] = (x, y, w, h)
        return cid

    def box(
        self,
        text: str,
        x: float,
        y: float,
        w: float,
        h: float,
        fill: str,
        stroke: str,
        *,
        fs: int = 15,
        bold: bool = False,
        dashed: bool = False,
        rounded: bool = True,
        align: str = "center",
        cid: str | None = None,
        extra: str = "",
    ) -> str:
        style = (
            f"rounded={1 if rounded else 0};arcSize=12;whiteSpace=wrap;html=1;"
            f"fillColor={fill};strokeColor={stroke};strokeWidth=1.6;"
            f"fontFamily={FONT};fontSize={fs};fontColor={COLORS['ink']};"
            f"fontStyle={1 if bold else 0};align={align};verticalAlign=middle;"
            f"spacingLeft=10;spacingRight=10;spacingTop=5;spacingBottom=5;"
            f"dashed={1 if dashed else 0};dashPattern=6 4;{extra}"
        )
        return self.vertex(esc(text), x, y, w, h, style, cid)

    def text(self, text: str, x: float, y: float, w: float, h: float, *, fs: int = 16, bold: bool = False, color: str | None = None, align: str = "left") -> str:
        style = (
            "text;html=1;whiteSpace=wrap;strokeColor=none;fillColor=none;"
            f"fontFamily={FONT};fontSize={fs};fontColor={color or COLORS['ink']};"
            f"fontStyle={1 if bold else 0};align={align};verticalAlign=middle;"
        )
        return self.vertex(esc(text), x, y, w, h, style)

    def frame(self, label: str, x: float, y: float, w: float, h: float, stroke: str, *, fill: str = "none", dashed: bool = True) -> str:
        style = (
            f"rounded=1;arcSize=10;whiteSpace=wrap;html=1;fillColor=none;strokeColor={stroke};"
            f"strokeWidth=1.8;dashed={1 if dashed else 0};dashPattern=7 5;"
            f"fontFamily={FONT};fontSize=15;fontStyle=1;fontColor={stroke};"
            "align=left;verticalAlign=top;spacingLeft=12;spacingTop=8;"
        )
        return self.vertex(esc(label), x, y, w, h, style)

    def edge(
        self,
        source: str,
        target: str,
        label: str = "",
        *,
        color: str | None = None,
        width: float = 2.0,
        dashed: bool = False,
        end: str = "block",
        extra: str = "",
        points: list[tuple[float, float]] | None = None,
    ) -> str:
        cid = self._new_id("e")
        port_style = ""
        if source != target and source in self.geometry and target in self.geometry:
            sx, sy, sw, sh = self.geometry[source]
            tx, ty, tw, th = self.geometry[target]
            dx = (tx + tw / 2) - (sx + sw / 2)
            dy = (ty + th / 2) - (sy + sh / 2)
            # Cross-column links should stay horizontal even when the target is
            # somewhat higher/lower; within a column, use top-to-bottom ports.
            if abs(dx) > 250:
                port_style = (
                    "exitX=1;exitY=0.5;entryX=0;entryY=0.5;"
                    if dx > 0 else
                    "exitX=0;exitY=0.5;entryX=1;entryY=0.5;"
                )
            elif dy >= 0:
                port_style = "exitX=0.5;exitY=1;entryX=0.5;entryY=0;"
            else:
                port_style = "exitX=0.5;exitY=0;entryX=0.5;entryY=1;"
        style = (
            "edgeStyle=orthogonalEdgeStyle;rounded=1;orthogonalLoop=1;jettySize=auto;html=1;"
            f"strokeColor={color or COLORS['line']};strokeWidth={width};"
            f"endArrow={end};endFill=1;endSize=8;dashed={1 if dashed else 0};dashPattern=6 4;"
            f"fontFamily={FONT};fontSize=13;fontColor={COLORS['muted']};labelBackgroundColor=#FFFFFF;"
            f"{port_style}{extra}"
        )
        cell = ET.Element("mxCell", {"id": cid, "value": esc(label), "style": style, "edge": "1", "parent": "1", "source": source, "target": target})
        geometry = ET.SubElement(cell, "mxGeometry", {"relative": "1", "as": "geometry"})
        if points:
            array = ET.SubElement(geometry, "Array", {"as": "points"})
            for px, py in points:
                ET.SubElement(array, "mxPoint", {"x": str(px), "y": str(py)})
        self.cells.append(cell)
        return cid

    def header(self, number: str) -> None:
        self.box(number, 42, 30, 58, 58, COLORS["deepgen"], COLORS["deepgen_s"], fs=24, bold=True, rounded=True)
        self.text(self.title, 120, 26, 1220, 42, fs=28, bold=True)
        self.text(self.subtitle, 120, 68, 1350, 30, fs=15, color=COLORS["muted"])
        self.box("蓝=输入  绿=编码  紫=推理\n橙=Adapter  红=Detail  灰=冻结", 1490, 30, 365, 58, COLORS["white"], "#B8C5D0", fs=12)

    def save(self, filename: str) -> None:
        mxfile = ET.Element("mxfile", {"host": "Electron", "modified": "2026-09-22T00:00:00.000Z", "agent": "Codex", "version": "31.4.5", "type": "device", "compressed": "false"})
        page = ET.SubElement(mxfile, "diagram", {"id": filename, "name": "Page-1"})
        model = ET.SubElement(page, "mxGraphModel", {
            "dx": "1422", "dy": "794", "grid": "1", "gridSize": "10", "guides": "1", "tooltips": "1",
            "connect": "1", "arrows": "1", "fold": "1", "page": "1", "pageScale": "1",
            "pageWidth": str(self.width), "pageHeight": str(self.height), "math": "0", "shadow": "0",
        })
        root = ET.SubElement(model, "root")
        ET.SubElement(root, "mxCell", {"id": "0"})
        ET.SubElement(root, "mxCell", {"id": "1", "parent": "0"})
        for cell in self.cells:
            root.append(cell)
        ET.indent(mxfile, space="  ")
        ET.ElementTree(mxfile).write(OUT / filename, encoding="utf-8", xml_declaration=True)


def overall_architecture() -> None:
    d = Diagram("项目整体架构", "当前 V6.4：V6.3 全局几何/交互控制 + 新增共享 Face-Hand Detail Branch", height=1220)
    d.header("01")

    d.frame("A  数据与离线资产", 35, 125, 330, 820, COLORS["input_s"], fill="#F7FBFF")
    datasets = d.box("数据集\niPER · Hi4D · 项目 TikTok", 65, 185, 270, 80, COLORS["input"], COLORS["input_s"], bold=True)
    source = d.box("源人物图 / 目标帧\n文本编辑指令", 65, 295, 270, 72, COLORS["input"], COLORS["input_s"], bold=True)
    preprocess = d.box("离线预处理\nSMPLest-X 参数\nDWPose 134 点\nnvdiffrast assets\nnormal / part / depth / mask", 65, 410, 270, 150, COLORS["encode"], COLORS["encode_s"], bold=True)
    base_assets = d.box("V6.3 全局资产\n25 pose heatmaps · 14 part maps\nnormal / depth / human mask\nSMPL-X global 26", 65, 600, 270, 128, COLORS["encode"], COLORS["encode_s"])
    detail_assets = d.box("V6.4 Detail 资产\nface 68 · hands 2×21\nSMPL-X detail 103 · ROI boxes\nface/hand reference crops 224²", 65, 765, 270, 140, COLORS["detail"], COLORS["detail_s"])
    d.edge(datasets, preprocess)
    d.edge(source, preprocess)
    d.edge(preprocess, base_assets)
    d.edge(preprocess, detail_assets)

    d.frame("B  运行时输入契约", 395, 125, 355, 820, COLORS["input_s"], fill="#F7FBFF")
    cb = d.box("ConditionBundle\nA/B 空间条件 + global tokens\nrelative geometry + contact + task", 425, 190, 295, 130, COLORS["input"], COLORS["input_s"], bold=True)
    ident = d.box("AdapterIdentityCondition\nsource scene latent [B,16,h,w]\nsource person latents [B,2,16,h,w]\nsource_indices / person_valid", 425, 365, 295, 140, COLORS["input"], COLORS["input_s"], bold=True)
    detail = d.box("FaceHandDetailCondition\nlandmarks + SMPL-X detail 103\nsource/target boxes + region_valid\nDetailReferenceBatch\n[B,2,3,R,3,224,224]", 425, 550, 295, 170, COLORS["detail"], COLORS["detail_s"], bold=True)
    noise = d.box("DeepGen 每步输入\nnoisy target latent · timestep\ntext hidden / pooled projection", 425, 770, 295, 105, COLORS["input"], COLORS["input_s"], bold=True)
    d.edge(base_assets, cb)
    d.edge(source, ident)
    d.edge(detail_assets, detail)

    d.frame("C  UnifiedSMPLXAdapterV6", 780, 125, 650, 820, COLORS["adapter_s"], fill="#FFFCF7")
    inject = d.box("SMPLXConditionInjector\nSpatialConditionEncoder\n+ token encoders", 815, 180, 265, 90, COLORS["encode"], COLORS["encode_s"], bold=True)
    reason = d.box("SMPLXAdapterReasoner\n单人几何 · 双人 cross-person\ncontact fusion", 815, 315, 265, 100, COLORS["reason"], COLORS["reason_s"], bold=True)
    bridge = d.box("ReasonerControlBridge\n+ Identity Tokens\ngeometry / interaction scene 128ch\n4 geometry + 8 appearance / person", 815, 460, 265, 120, COLORS["adapter"], COLORS["adapter_s"], bold=True)
    detailprep = d.box("FaceHandDetailPreparer\nFace encoder\n+ shared Hand encoder\nROI warp + local appearance tokens", 1125, 180, 265, 120, COLORS["detail"], COLORS["detail_s"], bold=True)
    core = d.box("SharedRecurrentControlCore\nGeometry Branch\nInteraction Branch（仅双人）\nDetail Branch（脸/手共享）\n6-stage recurrent injection", 960, 635, 320, 175, COLORS["adapter"], COLORS["adapter_s"], fs=15, bold=True)
    strength = d.box("ControlStrengthController\nbranch strength × group gate × schedule\nface/hand region gate", 960, 845, 320, 75, COLORS["adapter"], COLORS["adapter_s"], bold=True)
    d.edge(cb, inject)
    d.edge(inject, reason)
    d.edge(reason, bridge)
    d.edge(ident, bridge)
    d.edge(detail, detailprep, points=[(760, 610), (1100, 610), (1100, 240)])
    d.edge(bridge, core, "global context + 144ch conditions")
    d.edge(detailprep, core, "detail-only context + 144ch")
    d.edge(noise, core, "per denoise step")
    d.edge(core, strength, "6× residual groups")

    d.frame("D  冻结 DeepGen 运行时", 1460, 125, 405, 820, COLORS["deepgen_s"], fill="#F8FCF7")
    hook = d.box("ControlledDeepGenPipeline\nTransformer forward pre-hook", 1500, 190, 325, 72, COLORS["deepgen"], COLORS["deepgen_s"], bold=True)
    frozen = d.box("冻结组件\nVAE · LMM/VLM · Connector\nDeepGen Transformer", 1500, 305, 325, 105, COLORS["frozen"], COLORS["frozen_s"], bold=True, dashed=True)
    dit = d.box("DeepGen DiT: 24 blocks\n6 组残差注入 blocks 0–22\nblock 23 context-pre-only，不注入", 1500, 455, 325, 120, COLORS["deepgen"], COLORS["deepgen_s"], bold=True)
    sched = d.box("Flow-Match Euler Scheduler\n默认 28 denoise steps", 1500, 625, 325, 75, COLORS["deepgen"], COLORS["deepgen_s"])
    output = d.box("VAE Decode\n输出姿态编辑图像", 1500, 760, 325, 88, COLORS["deepgen"], COLORS["deepgen_s"], bold=True)
    d.edge(strength, hook, "block_controlnet_hidden_states", points=[(1445, 882), (1445, 226)])
    d.edge(hook, frozen, "调用冻结运行时")
    d.edge(frozen, dit, "提供冻结权重", dashed=True)
    d.edge(dit, sched)
    d.edge(sched, output)

    d.frame("E  训练、检查点与验收（底部横向链路）", 35, 980, 1830, 185, COLORS["muted"], fill="#FBFCFD")
    v63 = d.box("V6.3 checkpoint\ngeometry + interaction", 85, 1035, 250, 82, COLORS["frozen"], COLORS["frozen_s"], bold=True)
    train = d.box("iPER detail 单样本训练入口\ntrain_iper_v6_detail.py\n冻结 DeepGen 与旧 V6.3，仅训练 detail", 430, 1020, 390, 110, COLORS["detail"], COLORS["detail_s"], bold=True)
    v64 = d.box("V6.4 checkpoint\nbranch_names + detail_schema_version", 915, 1035, 300, 82, COLORS["adapter"], COLORS["adapter_s"], bold=True)
    evalb = d.box("验证\nDetail ON/OFF · frozen SHA-256\nFace NME · Hand PCK · PSNR/SSIM", 1310, 1020, 430, 110, COLORS["warn"], COLORS["warn_s"], bold=True)
    d.edge(v63, train)
    d.edge(train, v64)
    d.edge(v64, evalb)
    d.text("状态说明：模型核心已在 main；当前远端 iPER detail 数据集、条件构建与过拟合训练脚本仍是工作区实验文件。", 80, 1140, 1620, 24, fs=15, color=COLORS["muted"], align="center")
    d.save("01_project_overall_architecture.drawio")


def global_adapter() -> None:
    d = Diagram("全局 Adapter 整体架构（含 Encoder）", "这里的“全局”指 V6.3 Geometry + Interaction 控制链，不含 Face-Hand Detail Branch", height=1210)
    d.header("02")

    d.frame("1  原始条件", 35, 125, 330, 880, COLORS["input_s"], fill="#F7FBFF")
    maps = d.box("每个人 A / B\nnormal [3,H,W]\npose heatmaps [25,H,W]\npart one-hot [14,H,W]\nhuman mask [1,H,W]\noptional depth [1,H,W]", 70, 180, 260, 190, COLORS["input"], COLORS["input_s"], bold=True)
    globals_ = d.box("每个人全局向量\nSMPL-X global [26]", 70, 405, 260, 70, COLORS["input"], COLORS["input_s"], bold=True)
    dual = d.box("双人条件\nrelative geometry [11]\ncontact raster [2,H,W]\n≤8 structured contact relations\ntask_id", 70, 515, 260, 145, COLORS["input"], COLORS["input_s"], bold=True)
    identity = d.box("身份/外观\nsource scene latent [16,h,w]\nsource person latent [2,16,h,w]\nsource_indices [2]", 70, 700, 260, 125, COLORS["input"], COLORS["input_s"], bold=True)
    target = d.box("目标噪声 latent + text context\ntimestep / pooled text", 70, 865, 260, 92, COLORS["input"], COLORS["input_s"], bold=True)

    d.frame("2  Condition Injector / Encoders", 395, 125, 465, 880, COLORS["encode_s"], fill="#F6FCFA")
    stems = d.box("SpatialConditionEncoder\nnormal: 3→32→64→128\npose: 25→32→64→96→128\npart: 14→32→64→96→128\ndepth: 1→32→64→128（可选）", 430, 180, 395, 175, COLORS["encode"], COLORS["encode_s"], bold=True)
    fusion = d.box("concat → 1×1 conv 256\nconv + GN/SiLU + 2×ResBlock\nRoleFiLM(A/B)\n输出 [B,256,H/8,W/8]", 430, 390, 395, 130, COLORS["encode"], COLORS["encode_s"], bold=True)
    toks = d.box("Token Encoders（dim=256）\nGlobal: 26→4 tokens/person\nRelative: 11→2 tokens\nContact relation: ≤8 tokens\nTask: 1 token\nContact raster: 2→128ch @ H/8", 430, 560, 395, 175, COLORS["encode"], COLORS["encode_s"], bold=True)
    bundle = d.box("ConditionBundle\nperson_valid / person_count / masks", 430, 790, 395, 82, COLORS["encode"], COLORS["encode_s"], bold=True)
    d.edge(maps, stems)
    d.edge(stems, fusion)
    d.edge(globals_, toks)
    d.edge(dual, toks)
    d.edge(fusion, bundle)
    d.edge(toks, bundle)

    d.frame("3  SMPL-X Reasoning", 890, 125, 440, 880, COLORS["reason_s"], fill="#FBF8FE")
    person = d.box("PersonGeometryReasoner × A/B\nlocal spatial reasoning + global attention\nrole embedding", 925, 185, 370, 105, COLORS["reason"], COLORS["reason_s"], bold=True)
    cross = d.box("BidirectionalCrossPersonReasoner\n2 layers · 8 heads\n仅 person_count=2 的样本", 925, 340, 370, 100, COLORS["reason"], COLORS["reason_s"], bold=True)
    contact = d.box("ContactConditionReasoner\ncontact raster + relation token attention\ncontact gate init = -4", 925, 490, 370, 105, COLORS["reason"], COLORS["reason_s"], bold=True)
    fuse = d.box("DualFeatureFusion\n单人：local geometry path\n双人：geometry + cross-person + contact", 925, 650, 370, 115, COLORS["reason"], COLORS["reason_s"], bold=True)
    state = d.box("InternalControlState\ngeometry low [512,H/16,W/16]\n+ high [512,H/8,W/8]\ninteraction 同形状（单人严格为 0）\nperson tokens [B,2,H/16×W/16,512]", 925, 815, 370, 140, COLORS["reason"], COLORS["reason_s"], bold=True)
    d.edge(bundle, person)
    d.edge(person, cross)
    d.edge(cross, contact)
    d.edge(contact, fuse)
    d.edge(fuse, state)

    d.frame("4  Bridge + Global Control Core", 1360, 125, 505, 880, COLORS["adapter_s"], fill="#FFFCF7")
    bridge = d.box("ReasonerControlBridge\nlow + downsample(high) → 1×1 projection\ngeometry scene 128ch / interaction scene 128ch", 1395, 180, 435, 115, COLORS["adapter"], COLORS["adapter_s"], bold=True)
    cond = d.box("控制图拼接 @ target latent size\nGeometry: source scene 16 + geometry 128 = 144ch\nInteraction: zero scene 16 + interaction 128 = 144ch", 1395, 335, 435, 120, COLORS["adapter"], COLORS["adapter_s"], bold=True)
    idtokens = d.box("Global identity tokens（dim=1536）\nreasoner person tokens pool 2×2\n→ 4 geometry tokens/person\nsource person VAE latent pool 2×4\n→ 8 appearance tokens/person\nPersonTokenBinder = slot + source-index\n最多 2×12 = 24 tokens", 1395, 485, 435, 180, COLORS["adapter"], COLORS["adapter_s"], bold=True)
    branches = d.box("SharedRecurrentControlCore\nGeometry Branch：全样本\nInteraction Branch：仅双人有效行\n各自 condition embed / state\n6×rank-64 adapter / 6×zero head\n共享一个 DeepGen joint transformer block", 1395, 700, 435, 165, COLORS["adapter"], COLORS["adapter_s"], bold=True)
    residuals = d.box("输出\ngeometry 6×[B,Ntarget,1536]\ninteraction 6×[B,Ntarget,1536]", 1395, 905, 435, 70, COLORS["deepgen"], COLORS["deepgen_s"], bold=True)
    d.edge(state, bridge, points=[(1335, 885), (1335, 237)])
    d.edge(bridge, cond)
    d.edge(identity, cond, points=[(360, 762), (870, 762), (870, 990), (1370, 990), (1370, 395)])
    d.edge(state, idtokens, points=[(1335, 885), (1335, 575)])
    d.edge(identity, idtokens, points=[(360, 762), (870, 762), (870, 990), (1370, 990), (1370, 575)])
    d.edge(cond, branches, points=[(1380, 470), (1380, 782)])
    d.edge(idtokens, branches, "text context + adapter tokens")
    d.edge(target, branches, points=[(360, 911), (870, 911), (870, 1000), (1370, 1000), (1370, 782)])
    d.edge(branches, residuals)

    d.frame("关键边界", 35, 1035, 1830, 120, COLORS["warn_s"], fill="#FFFBEE")
    d.text("Geometry Branch 负责全局姿态/外观布局；Interaction Branch 只处理双人关系。Face/Hand 的 68/21 点与 SMPL-X detail 103 不进入这条全局链，而由独立 Detail Preparer 编码后进入共享 Detail Branch。", 80, 1065, 1735, 56, fs=15, bold=True, align="center")
    d.save("02_global_adapter_with_encoders.drawio")


def face_adapter() -> None:
    d = Diagram("Face Adapter 架构（含 Encoder）", "当前代码中的 Face-specific encoder + 共享 FaceHandDetailBranch；并不存在独立 Face 控制分支", height=1190)
    d.header("03")

    d.frame("A  源脸外观路径", 35, 135, 490, 725, COLORS["detail_s"], fill="#FFF9FA")
    crop = d.box("源人物脸部 ROI\nsource face box → crop 224×224\n支持 R=1…3，reference_valid", 75, 195, 410, 105, COLORS["input"], COLORS["input_s"], bold=True)
    stats = d.box("当前默认外观特征\nRGB mean(3) + sqrt variance(3) = 6D", 75, 350, 410, 82, COLORS["detail"], COLORS["detail_s"], bold=True)
    provider = d.box("可选 ReferenceFeatureProvider\n外部预计算特征接口\nArcFace / DINO / SigLIP 尚未接入主运行时", 75, 480, 410, 105, COLORS["warn"], COLORS["warn_s"], dashed=True, bold=True)
    appearance = d.box("DetailAppearanceEncoder\nLN(6) → Linear256 → SiLU\n→ Linear(8×1536)\n多参考按 valid 平均；缺失使用 face null tokens", 75, 635, 410, 145, COLORS["detail"], COLORS["detail_s"], bold=True)
    d.edge(crop, stats)
    d.edge(stats, appearance)
    d.edge(provider, appearance, "可替换 6D reference_features", dashed=True)

    d.frame("B  目标脸几何路径", 555, 135, 510, 725, COLORS["encode_s"], fill="#F8FCFA")
    geom_in = d.box("目标脸条件 / person\nface_keypoints 68×(x,y,confidence) = 204\nexpression 10 + jaw pose 3\n合计 217D", 595, 195, 430, 115, COLORS["input"], COLORS["input_s"], bold=True)
    faceenc = d.box("Face Encoder\nLayerNorm217 → Linear256 → SiLU → Linear128\n输出 face region feature [128]", 595, 360, 430, 105, COLORS["encode"], COLORS["encode_s"], bold=True)
    mask = d.box("target face box + region_valid\nsoft_box_mask @ latent / token resolution\ninvalid 区域严格为 0", 595, 515, 430, 95, COLORS["encode"], COLORS["encode_s"], bold=True)
    paint = d.box("区域散射 / 加权平均\n128D face feature × target mask\n写入 detail geometry canvas [B,128,Hlat,Wlat]", 595, 660, 430, 120, COLORS["encode"], COLORS["encode_s"], bold=True)
    d.edge(geom_in, faceenc)
    d.edge(faceenc, paint)
    d.edge(mask, paint)

    d.frame("C  Face Detail 条件组装与控制", 1095, 135, 770, 725, COLORS["adapter_s"], fill="#FFFCF7")
    latentwarp = d.box("源 VAE latent ROI warp\nsource face box → target face box\ngrid_sample，得到 16ch source canvas", 1135, 190, 300, 105, COLORS["adapter"], COLORS["adapter_s"], bold=True)
    cond = d.box("Detail spatial condition\n16ch source face canvas\n+ 128ch geometry\n= [B,144,Hlat,Wlat]", 1135, 350, 300, 100, COLORS["adapter"], COLORS["adapter_s"], bold=True)
    bind = d.box("Face token binding\n8 appearance tokens\n+ person/source binding\n+ region(face) + side(face) embedding\n→ [B,8,1536] / person", 1500, 190, 325, 130, COLORS["detail"], COLORS["detail_s"], bold=True)
    context = d.box("独立 Detail Context\ntext context + detail tokens\n不改变 Geometry / Interaction context", 1500, 375, 325, 95, COLORS["detail"], COLORS["detail_s"], bold=True)
    branch = d.box("共享 FaceHandDetailBranch（6 stages）\ncondition embed 144→1536 tokens\nshared joint block + rank-64 stage adapter\nCrossNorm + zero-init head", 1280, 535, 390, 140, COLORS["adapter"], COLORS["adapter_s"], bold=True)
    gate = d.box("Face Gate\nface soft mask × face_strength\n× detail_strength × cosine schedule\n(progress 0.35→1) × group gate", 1280, 715, 390, 110, COLORS["detail"], COLORS["detail_s"], bold=True)
    d.edge(paint, cond, points=[(1080, 720), (1080, 400)])
    d.edge(latentwarp, cond)
    d.edge(appearance, bind, points=[(530, 840), (1840, 840), (1840, 255)])
    d.edge(bind, context)
    d.edge(cond, branch)
    d.edge(context, branch)
    d.edge(branch, gate, "6× face-aware residuals")

    d.frame("D  注入 DeepGen DiT", 35, 900, 1830, 215, COLORS["deepgen_s"], fill="#F8FCF7")
    align = d.box("target-token residual\n[B,Ntarget,1536]", 110, 965, 265, 78, COLORS["adapter"], COLORS["adapter_s"], bold=True)
    pad = d.box("align_target_residuals\n为 source-reference / padding tokens 补 0\n→ [B,Nfull,1536]", 505, 950, 345, 108, COLORS["adapter"], COLORS["adapter_s"], bold=True)
    dit = d.box("DeepGen 24-block DiT\n注入 blocks 0–22（6 组）\nblock 23 不注入", 995, 950, 340, 108, COLORS["deepgen"], COLORS["deepgen_s"], bold=True)
    out = d.box("VAE Decode\n身份与局部脸部细节增强", 1490, 965, 285, 78, COLORS["deepgen"], COLORS["deepgen_s"], bold=True)
    d.edge(gate, align, points=[(1475, 875), (242, 875)])
    d.edge(align, pad)
    d.edge(pad, dit)
    d.edge(dit, out)
    d.text("实现边界：当前 Face encoder 使用 landmarks + expression/jaw；外观 token 默认只来自 ROI 像素统计。ArcFace/DINO/SigLIP 是预留 provider 的下一步，不应视为已实现。", 150, 1080, 1600, 28, fs=15, color=COLORS["muted"], align="center")
    d.save("03_face_adapter_with_encoder.drawio")


def hand_adapter() -> None:
    d = Diagram("Hand Adapter 架构（含 Encoder）", "左右手共享一个 Hand encoder；左手先规范化到右手坐标系，控制核心与 Face 共用 Detail Branch", height=1190)
    d.header("04")

    d.frame("A  目标手部几何编码", 35, 135, 640, 720, COLORS["encode_s"], fill="#F8FCFA")
    left = d.box("左手条件\n21×(x,y,confidence)\n+ 45D SMPL-X hand pose", 75, 195, 255, 95, COLORS["input"], COLORS["input_s"], bold=True)
    canon = d.box("左手规范化\nx → 1-x\naxis-angle [rx,ry,rz]\n→ [rx,-ry,-rz]", 375, 195, 255, 95, COLORS["warn"], COLORS["warn_s"], bold=True)
    right = d.box("右手条件\n21×(x,y,confidence) + 45D pose\n保持原坐标", 75, 345, 255, 95, COLORS["input"], COLORS["input_s"], bold=True)
    shared = d.box("共享 Hand Encoder\n输入 21×3 +45 =108D\nLayerNorm108 → Linear256\n→ SiLU → Linear128", 375, 345, 255, 110, COLORS["encode"], COLORS["encode_s"], bold=True)
    masks = d.box("left/right target boxes\n+ region_valid\nsoft masks @ latent/token", 75, 520, 255, 95, COLORS["encode"], COLORS["encode_s"], bold=True)
    canvas = d.box("局部几何 canvas\n左/右 128D region feature\n× mask\n多区域按权重平均", 375, 520, 255, 120, COLORS["encode"], COLORS["encode_s"], bold=True)
    geomout = d.box("Hand geometry contribution\n[B,128,Hlat,Wlat]", 220, 700, 270, 75, COLORS["encode"], COLORS["encode_s"], bold=True)
    d.edge(left, canon)
    d.edge(canon, shared)
    d.edge(right, shared)
    d.edge(shared, canvas)
    d.edge(masks, canvas)
    d.edge(canvas, geomout)

    d.frame("B  源手外观与绑定", 705, 135, 535, 720, COLORS["detail_s"], fill="#FFF9FA")
    refs = d.box("源左手 / 右手 ROI crops\n每侧 R=1…3，224×224\nsource boxes + reference_valid", 745, 195, 455, 100, COLORS["input"], COLORS["input_s"], bold=True)
    feat = d.box("当前默认局部外观特征\n每 crop: RGB mean + sqrt variance = 6D", 745, 345, 455, 80, COLORS["detail"], COLORS["detail_s"], bold=True)
    app = d.box("共享 DetailAppearanceEncoder\n6→256→8×1536 tokens / hand\n缺失源手 → anatomical null appearance token", 745, 475, 455, 110, COLORS["detail"], COLORS["detail_s"], bold=True)
    binder = d.box("DetailTokenBinder\nperson/source binding + region embedding\n+ anatomical side embedding(left/right)\n固定键 = (source_index, person_slot, side)", 745, 635, 455, 120, COLORS["detail"], COLORS["detail_s"], bold=True)
    d.edge(refs, feat)
    d.edge(feat, app)
    d.edge(app, binder)

    d.frame("C  Shared Detail Control", 1270, 135, 595, 720, COLORS["adapter_s"], fill="#FFFCF7")
    warp = d.box("源 hand VAE ROI warp\nsource left/right box → target left/right box\n16ch source canvas；重叠区域求平均", 1310, 190, 515, 105, COLORS["adapter"], COLORS["adapter_s"], bold=True)
    cond = d.box("detail_condition\n16ch source canvas + 128ch detail geometry\n= [B,144,Hlat,Wlat]", 1310, 350, 515, 95, COLORS["adapter"], COLORS["adapter_s"], bold=True)
    context = d.box("detail_tokens\n最多 2 people × 2 hands × 8 tokens\n作为独立 Detail context 的一部分", 1310, 500, 515, 95, COLORS["detail"], COLORS["detail_s"], bold=True)
    branch = d.box("FaceHandDetailBranch × 6 stages\n共享 DeepGen joint block\n每 stage 独立 rank-64 adapter / CrossNorm / zero head", 1310, 650, 515, 115, COLORS["adapter"], COLORS["adapter_s"], bold=True)
    d.edge(geomout, cond, points=[(650, 738), (650, 840), (1260, 840), (1260, 385)])
    d.edge(warp, cond)
    d.edge(binder, context)
    d.edge(cond, branch, points=[(1290, 470), (1290, 705)])
    d.edge(context, branch)

    d.frame("D  手部门控与 DiT 注入", 35, 900, 1830, 215, COLORS["deepgen_s"], fill="#F8FCF7")
    valid = d.box("有效性规则\n源手不可见：null token，目标几何仍可用\n目标手 invalid：区域残差严格为 0", 95, 945, 390, 112, COLORS["warn"], COLORS["warn_s"], bold=True)
    gate = d.box("Hand region gate\nmax(left mask, right mask)\n× hand_strength × detail_strength\n× cosine schedule × group gate", 600, 945, 390, 112, COLORS["detail"], COLORS["detail_s"], bold=True)
    inject = d.box("6 组 residuals 对齐到 Nfull\n注入 DeepGen blocks 0–22", 1110, 960, 310, 82, COLORS["adapter"], COLORS["adapter_s"], bold=True)
    result = d.box("输出\n手部结构/外观细节增强\n禁止跨人物或跨左右手借用", 1540, 950, 270, 102, COLORS["deepgen"], COLORS["deepgen_s"], bold=True)
    d.edge(branch, gate)
    d.edge(valid, gate)
    d.edge(gate, inject)
    d.edge(inject, result)
    d.text("说明：左右手只共享编码器参数，不共享绑定身份；side embedding 与 source/person binding 保留解剖侧和人物归属。", 250, 1080, 1400, 28, fs=15, color=COLORS["muted"], align="center")
    d.save("04_hand_adapter_with_encoder.drawio")


def input_conditions() -> None:
    d = Diagram("输入条件架构", "从 iPER/Hi4D/TikTok 的离线资产到 Adapter 的五类运行时输入对象", height=1230)
    d.header("05")

    d.frame("A  离线来源与切片", 35, 130, 430, 920, COLORS["input_s"], fill="#F7FBFF")
    smpl = d.box("SMPLest-X params.pt\n全局组：beta/root/trans/camera → 26D\n细节组：expression10 + jaw3\n+ L45 + R45 → 103D", 75, 185, 350, 125, COLORS["input"], COLORS["input_s"], bold=True)
    kp = d.box("DWPose / 项目 keypoints_134\n0:18 body · 18:24 foot\n24:92 face68\n92:113 left hand21\n113:134 right hand21", 75, 350, 350, 140, COLORS["input"], COLORS["input_s"], bold=True)
    render = d.box("nvdiffrast / geometry assets\nnormal3 · part14 · depth1 · mask1\n目标 pose heatmaps 25", 75, 530, 350, 105, COLORS["input"], COLORS["input_s"], bold=True)
    vae = d.box("DeepGen VAE encodes source images\nscene latent [B,16,h,w]\nperson latents [B,2,16,h,w]", 75, 675, 350, 100, COLORS["input"], COLORS["input_s"], bold=True)
    roi = d.box("ROI 条件构建\nface/hand box: xyxy normalized\nface expand15%，hand expand25%\nvalid threshold: face≥48/68, hand≥12/21\nreference crop 224²", 75, 815, 350, 155, COLORS["detail"], COLORS["detail_s"], bold=True)

    d.frame("B  五类输入对象", 500, 130, 740, 920, COLORS["encode_s"], fill="#F8FCFA")
    bundle = d.box("① ConditionBundle（全局结构）\nA/B spatial [256,H/8,W/8] + masks\nA/B global tokens [4,256]\nrelative tokens [2,256]\ncontact spatial/tokens + task token\nperson_valid / person_count", 540, 180, 310, 185, COLORS["encode"], COLORS["encode_s"], bold=True)
    identity = d.box("② AdapterIdentityCondition\n（全局外观）\nsource_person_latents [B,2,16,h,w]\nsource_indices [B,2] ∈ [0,15]\n同一样本有效人物不可重复", 890, 180, 310, 135, COLORS["encode"], COLORS["encode_s"], bold=True)
    detail = d.box("③ FaceHandDetailCondition（局部几何）\nface [B,2,68,3]\nhands [B,2,2,21,3]\nsmplx_detail [B,2,103]\nsource/target boxes [B,2,3,4]\nregion_valid [B,2,3] · source_indices", 540, 430, 310, 200, COLORS["detail"], COLORS["detail_s"], bold=True)
    refs = d.box("④ DetailReferenceBatch（局部外观）\nimages [B,2,3,R,3,224,224]\nregion order = face / left / right\nR=1…3 · reference_valid 同形槽位", 890, 430, 310, 150, COLORS["detail"], COLORS["detail_s"], bold=True)
    deep = d.box("⑤ DeepGen step inputs（动态）\ntarget noisy latent [B,16,Hlat,Wlat]\nencoder_hidden_states [B,Ntext,4096]\npooled_projections [B,2048]\ntimestep [B] / denoise progress", 715, 710, 310, 155, COLORS["input"], COLORS["input_s"], bold=True)
    strengths = d.box("运行时调节\ngeometry_strength\ninteraction_strength\ndetail / face / hand strength", 715, 905, 310, 92, COLORS["adapter"], COLORS["adapter_s"], bold=True)
    d.edge(smpl, bundle)
    d.edge(kp, bundle, "body 25")
    d.edge(render, bundle)
    d.edge(vae, identity, points=[(480, 725), (480, 110), (870, 110), (870, 247)])
    d.edge(smpl, detail)
    d.edge(kp, detail, "face + hands")
    d.edge(roi, detail, points=[(480, 897), (480, 530)])
    d.edge(roi, refs, points=[(480, 897), (480, 650), (870, 650), (870, 505)])

    d.frame("C  Adapter 内部准备结果", 1275, 130, 590, 920, COLORS["adapter_s"], fill="#FFFCF7")
    global_p = d.box("Prepared global conditioning\ngeometry_condition [B,144,Hlat,Wlat]\ninteraction_condition [B,144,Hlat,Wlat]\nadapter_tokens [B,≤24,1536]\nadapter_token_mask", 1320, 185, 500, 145, COLORS["adapter"], COLORS["adapter_s"], bold=True)
    detail_p = d.box("PreparedFaceHandDetailConditioning\ndetail_condition [B,144,Hlat,Wlat]\ndetail_tokens [B,48,1536]\ndetail_token_mask [B,48]\nregion_masks [B,2,3,Htok,Wtok]\ndetail_valid [B]", 1320, 400, 500, 175, COLORS["detail"], COLORS["detail_s"], bold=True)
    contexts = d.box("Context 隔离\nGeometry / Interaction: text + global adapter tokens\nDetail: text + detail tokens only\n局部 token 不改变旧分支输出", 1320, 650, 500, 120, COLORS["adapter"], COLORS["adapter_s"], bold=True)
    validation = d.box("关键校验\ndetail-valid person 必须属于全局 person_valid\ndetail source_indices 必须匹配 identity source_indices\n无 detail 或 detail_strength=0 → V6.3 路径", 1320, 835, 500, 125, COLORS["warn"], COLORS["warn_s"], bold=True)
    d.edge(bundle, global_p, points=[(860, 165), (1285, 165), (1285, 250)])
    d.edge(identity, global_p)
    d.edge(detail, detail_p, points=[(860, 650), (1285, 650), (1285, 487)])
    d.edge(refs, detail_p)
    d.edge(global_p, contexts)
    d.edge(detail_p, contexts)
    d.edge(deep, contexts)
    d.edge(strengths, validation)

    d.frame("维度解释与当前边界", 35, 1085, 1830, 92, COLORS["warn_s"], fill="#FFFBEE")
    d.text("当前代码不是把“137 维关键点”整体塞进一个 encoder：全局分支保留 25 个身体 pose heatmaps；局部分支单独取 68 脸点 + 2×21 手点，并与 SMPL-X detail 103D 编码。这样避免高维局部信号稀释全局姿态条件。", 85, 1105, 1730, 48, fs=15, bold=True, align="center")
    d.save("05_input_condition_architecture.drawio")


def dit_control() -> None:
    d = Diagram("Adapter 调控 DeepGen DiT 的架构", "每个 denoise step 通过 forward pre-hook 动态生成 6 组残差；不修改冻结的 DeepGen 源代码", height=1240)
    d.header("06")

    d.frame("A  每步输入与共享嵌入", 35, 130, 470, 930, COLORS["input_s"], fill="#F7FBFF")
    hook = d.box("ControlledDeepGenPipeline\ntransformer.register_forward_pre_hook", 75, 185, 390, 78, COLORS["deepgen"], COLORS["deepgen_s"], bold=True)
    step = d.box("step k / K\nprogress = k / (K-1)\ntimestep [B]", 75, 305, 390, 80, COLORS["input"], COLORS["input_s"], bold=True)
    target = d.box("target noisy latent [B,16,Hlat,Wlat]\nDeepGen patch embed (patch=2)\n→ target_hidden [B,Ntarget,1536]", 75, 430, 390, 105, COLORS["input"], COLORS["input_s"], bold=True)
    text = d.box("text hidden [B,Ntext,4096]\ncontext embedder → [B,Ntext,1536]\npooled text [B,2048] + timestep → temb", 75, 580, 390, 105, COLORS["input"], COLORS["input_s"], bold=True)
    contexts = d.box("分支 context\nGeometry: text + global adapter tokens\nInteraction: text + global adapter tokens\nDetail: text + detail tokens（独立）", 75, 735, 390, 115, COLORS["adapter"], COLORS["adapter_s"], bold=True)
    conditions = d.box("三份空间 condition\nGeometry 144ch · Interaction 144ch\nDetail 144ch\n各自独立 condition_embed → 1536", 75, 900, 390, 105, COLORS["adapter"], COLORS["adapter_s"], bold=True)
    d.edge(hook, step)
    d.edge(step, target)
    d.edge(step, text)
    d.edge(text, contexts)

    d.frame("B  SharedRecurrentControlCore：三分支 × 六级递归", 540, 130, 740, 930, COLORS["adapter_s"], fill="#FFFCF7")
    geo = d.box("Geometry Branch\n全样本运行\n独立 hidden/context\nstate", 580, 190, 190, 100, COLORS["adapter"], COLORS["adapter_s"], bold=True)
    inter = d.box("Interaction Branch\nindex_select\n双人有效行\n单人输出保持 0", 815, 190, 190, 100, COLORS["reason"], COLORS["reason_s"], bold=True)
    det = d.box("Detail Branch\n仅 detail_valid 行\nface / hand 共享", 1050, 190, 190, 100, COLORS["detail"], COLORS["detail_s"], bold=True)
    init = d.box("branch hidden0 = target_hidden + condition_embed(condition)\nbranch context0 = 对应 context", 655, 340, 510, 76, COLORS["adapter"], COLORS["adapter_s"], bold=True)
    stage = d.box("Stage s = 0…5\n① + stage_embedding[s]\n② 共享 DeepGen joint transformer block（复制自 block 0）\n③ + 独立 LowRankStageAdapter 1536→64→1536\n④ CrossNorm(control_hidden, target_hidden)\n⑤ zero-init Linear head → residual group s\n⑥ 更新本分支 hidden 与 context，进入下一 stage", 650, 475, 520, 225, COLORS["adapter"], COLORS["adapter_s"], fs=15, bold=True)
    shared = d.box("共享权重\npos_embed\ntime_text_embed\ncontext_embedder\n1× joint transformer block", 580, 760, 280, 105, COLORS["frozen"], COLORS["frozen_s"], dashed=True, bold=True)
    independent = d.box("分支独立参数\ncondition embed\n6 stage embeddings\n6 rank-64 adapters\n6 CrossNorm + 6 zero heads", 940, 760, 280, 120, COLORS["adapter"], COLORS["adapter_s"], bold=True)
    raw = d.box("BranchControlResiduals\ngeometry / interaction / optional detail\n各 6×[B,Ntarget,1536]", 720, 930, 390, 88, COLORS["adapter"], COLORS["adapter_s"], bold=True)
    d.edge(geo, init)
    d.edge(inter, init)
    d.edge(det, init)
    d.edge(target, init)
    d.edge(contexts, init)
    d.edge(conditions, init)
    d.edge(init, stage)
    d.edge(shared, stage, "每 stage 复用", dashed=True)
    d.edge(independent, stage)
    d.edge(stage, stage, "递归 ×6", color=COLORS["adapter_s"], extra="loopDirection=1;")
    d.edge(stage, raw)

    d.frame("C  强度、对齐与 24-block DiT 注入", 1315, 130, 550, 930, COLORS["deepgen_s"], fill="#F8FCF7")
    strength = d.box("ControlStrengthController / group s\nRₛ = Gₛ·geometry_strength·schedule\n + Iₛ·interaction_strength·schedule\n + Dₛ·region_gate·detail_strength·detail_schedule\n所有 group gate = exp(log_scale) > 0", 1355, 185, 470, 165, COLORS["adapter"], COLORS["adapter_s"], bold=True)
    regions = d.box("Detail region_gate\nface = max(person face masks) × face_strength\nhands = max(person,left/right masks) × hand_strength\nregion_gate = max(face, hands)\ndetail schedule: cosine, progress 0.35→1", 1355, 400, 470, 150, COLORS["detail"], COLORS["detail_s"], bold=True)
    align = d.box("align_target_residuals\n把 [B,Ntarget,1536] 对齐到 DeepGen 完整 token 顺序\nsource-reference 与 padding token 位置补 0\n→ 6×[B,Nfull,1536]", 1355, 605, 470, 135, COLORS["adapter"], COLORS["adapter_s"], bold=True)
    mapping = d.box("Active block mapping\nGroup 0 → blocks 0–3\nGroup 1 → blocks 4–7\nGroup 2 → blocks 8–11\nGroup 3 → blocks 12–15\nGroup 4 → blocks 16–19\nGroup 5 → blocks 20–22\nBlock 23: context_pre_only，不注入", 1355, 795, 470, 200, COLORS["deepgen"], COLORS["deepgen_s"], fs=15, bold=True)
    d.edge(raw, strength)
    d.edge(regions, strength)
    d.edge(strength, align)
    d.edge(align, mapping, "block_controlnet_hidden_states")

    d.frame("冻结 DeepGen 内部效果", 35, 1095, 1830, 92, COLORS["frozen_s"], fill="#F7F8FA")
    d.text("每个被映射的 DiT block 在自己的 hidden stream 中加上对应 residual；Adapter 通过 hook 提供控制，不改 Transformer 权重。detail 缺失或 detail_strength=0 时跳过 Detail Branch，保持 V6.3 行为。", 90, 1115, 1720, 48, fs=15, bold=True, align="center")
    d.save("06_adapter_controls_deepgen_dit.drawio")


def main() -> None:
    overall_architecture()
    global_adapter()
    face_adapter()
    hand_adapter()
    input_conditions()
    dit_control()
    print(f"Generated 6 editable draw.io diagrams in {OUT}")


if __name__ == "__main__":
    main()
