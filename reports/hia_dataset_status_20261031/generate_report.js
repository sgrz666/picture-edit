// HIA 脸手数据集构建状态报告 2026-10-31
// docx-js generator; follows documents:docx skill rules (R1 cover + DM-1, Profile A formal body,
// 3-section page numbering, TOC with placeholders, horizontal-only tables, image aspect preserved).
const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
  ImageRun, PageBreak, Header, Footer, PageNumber, NumberFormat,
  AlignmentType, HeadingLevel, WidthType, BorderStyle, ShadingType,
  TableLayoutType, SectionType, TableOfContents,
} = require("docx");
const fs = require("fs");
const path = require("path");

const ASSETS = path.join(__dirname, "assets");
const OUT = path.join(__dirname, "HIA脸手数据集构建状态报告_20261031.docx");

// ── palette: DM-1 cover tokens + darkened body tokens ──
const P = {
  bg: "162235", accent: "37DCF2",
  cover: { titleColor: "FFFFFF", subtitleColor: "B0B8C0", metaColor: "90989F", footerColor: "687078" },
  primary: "14324A", body: "000000", secondary: "5A6B7A",
  table: { headerBg: "1B6B7A", headerText: "FFFFFF", innerLine: "C8DDE2", surface: "EDF3F5" },
};

const NB = { style: BorderStyle.NONE, size: 0, color: "FFFFFF" };
const noBorders = { top: NB, bottom: NB, left: NB, right: NB };
const allNoBorders = { top: NB, bottom: NB, left: NB, right: NB, insideHorizontal: NB, insideVertical: NB };

// ── cover layout helpers (from design-system.md) ──
function splitTitleLines(title, charsPerLine) {
  if (title.length <= charsPerLine) return [title];
  const breakAfter = new Set([..."，。、；：！？", ..."的与和及之在于为", ..."-_—–·/", ..." \t"]);
  const lines = [];
  let remaining = title;
  while (remaining.length > charsPerLine) {
    let breakAt = -1;
    for (let i = charsPerLine; i >= Math.floor(charsPerLine * 0.6); i--) {
      if (i < remaining.length && breakAfter.has(remaining[i - 1])) { breakAt = i; break; }
    }
    if (breakAt === -1) {
      const limit = Math.min(remaining.length, Math.ceil(charsPerLine * 1.3));
      for (let i = charsPerLine + 1; i < limit; i++) {
        if (breakAfter.has(remaining[i - 1])) { breakAt = i; break; }
      }
    }
    if (breakAt === -1) {
      breakAt = charsPerLine;
      const prevChar = remaining[breakAt - 1], nextChar = remaining[breakAt];
      if (prevChar && nextChar && !breakAfter.has(prevChar) && !breakAfter.has(nextChar) &&
          /[\u4e00-\u9fff]/.test(prevChar) && /[\u4e00-\u9fff]/.test(nextChar)) breakAt -= 1;
    }
    lines.push(remaining.slice(0, breakAt).trim());
    remaining = remaining.slice(breakAt).trim();
  }
  if (remaining) lines.push(remaining);
  if (lines.length > 1 && lines[lines.length - 1].length <= 2) {
    const last = lines.pop();
    lines[lines.length - 1] += last;
  }
  return lines;
}

function calcTitleLayout(title, maxWidthTwips, preferredPt = 40, minPt = 24) {
  const charsPerLine = (pt) => Math.floor(maxWidthTwips / (pt * 20));
  let titlePt = preferredPt, lines;
  while (titlePt >= minPt) {
    const cpl = charsPerLine(titlePt);
    if (cpl < 2) { titlePt -= 2; continue; }
    lines = splitTitleLines(title, cpl);
    if (lines.length <= 3) break;
    titlePt -= 2;
  }
  if (!lines || lines.length > 3) {
    lines = splitTitleLines(title, charsPerLine(minPt));
    titlePt = minPt;
  }
  return { titlePt, titleLines: lines };
}

function calcCoverSpacing(params) {
  const { titleLineCount = 1, titlePt = 36, hasSubtitle = false, hasEnglishLabel = false,
          metaLineCount = 0, fixedHeight = 800, pageHeight = 16838 } = params;
  const SAFETY = 1200;
  const usableHeight = pageHeight - SAFETY;
  const titleHeight = titleLineCount * (titlePt * 23 + 200);
  const subtitleHeight = hasSubtitle ? (12 * 23 + 600) : 0;
  const englishLabelHeight = hasEnglishLabel ? (9 * 23 + 600) : 0;
  const metaHeight = metaLineCount * (10 * 23 + 100);
  const implicitParaHeight = 3 * 300;
  const contentHeight = titleHeight + subtitleHeight + englishLabelHeight + metaHeight + fixedHeight + implicitParaHeight;
  const safeRemaining = Math.max(usableHeight - contentHeight, 400);
  const FOOTER_MIN = 800;
  const rawTop = Math.floor(safeRemaining * 0.45);
  const rawBottom = Math.floor(safeRemaining * 0.45);
  const bottomSpacing = Math.max(rawBottom, FOOTER_MIN);
  const topSpacing = Math.max(rawTop - Math.max(0, FOOTER_MIN - rawBottom), 400);
  const midSpacing = Math.max(safeRemaining - topSpacing - bottomSpacing, 0);
  return { topSpacing, midSpacing, bottomSpacing };
}

// ── cover recipe R1 (design-system.md, verbatim structure) ──
function buildCoverR1(config) {
  const C = config.palette;
  const padL = 1200, padR = 800;
  const availableWidth = 11906 - padL - padR - 300;
  const { titlePt, titleLines } = calcTitleLayout(config.title, availableWidth, 40, 24);
  const titleSize = titlePt * 2;
  const spacing = calcCoverSpacing({
    titleLineCount: titleLines.length, titlePt,
    hasSubtitle: !!config.subtitle, hasEnglishLabel: !!config.englishLabel,
    metaLineCount: (config.metaLines || []).length, fixedHeight: 400,
  });
  const accentLeft = { style: BorderStyle.SINGLE, size: 8, color: C.accent, space: 12 };
  const children = [];
  children.push(new Paragraph({ spacing: { before: spacing.topSpacing } }));
  if (config.englishLabel) {
    children.push(new Paragraph({
      indent: { left: padL, right: padR }, spacing: { after: 500 },
      border: { bottom: { style: BorderStyle.SINGLE, size: 6, color: C.accent, space: 8 } },
      children: [new TextRun({ text: config.englishLabel.split("").join("  "),
        size: 18, color: C.accent, font: { ascii: "Calibri", eastAsia: "SimHei" }, characterSpacing: 40 })],
    }));
  }
  for (let i = 0; i < titleLines.length; i++) {
    children.push(new Paragraph({
      indent: { left: padL },
      spacing: { after: i < titleLines.length - 1 ? 100 : 300, line: Math.ceil(titlePt * 23), lineRule: "atLeast" },
      children: [new TextRun({ text: titleLines[i], size: titleSize, bold: true,
        color: C.cover.titleColor, font: { eastAsia: "SimHei", ascii: "Arial" } })],
    }));
  }
  if (config.subtitle) {
    children.push(new Paragraph({
      indent: { left: padL }, spacing: { after: 800 },
      children: [new TextRun({ text: config.subtitle, size: 24, color: C.cover.subtitleColor,
        font: { eastAsia: "Microsoft YaHei", ascii: "Arial" } })],
    }));
  }
  for (const line of (config.metaLines || [])) {
    children.push(new Paragraph({
      indent: { left: padL + 200 }, spacing: { after: 80 },
      border: { left: accentLeft },
      children: [new TextRun({ text: line, size: 24, color: C.cover.metaColor,
        font: { eastAsia: "Microsoft YaHei", ascii: "Arial" } })],
    }));
  }
  children.push(new Paragraph({ spacing: { before: spacing.bottomSpacing } }));
  children.push(new Paragraph({
    indent: { left: padL, right: padR },
    border: { top: { style: BorderStyle.SINGLE, size: 2, color: C.accent, space: 8 } },
    spacing: { before: 200 },
    children: [
      new TextRun({ text: config.footerLeft || "", size: 16, color: C.cover.footerColor, font: { ascii: "Arial", eastAsia: "Microsoft YaHei" } }),
      new TextRun({ text: "                                        " }),
      new TextRun({ text: config.footerRight || "", size: 16, color: C.cover.footerColor, font: { ascii: "Arial", eastAsia: "Microsoft YaHei" } }),
    ],
  }));
  return [new Table({
    width: { size: 100, type: WidthType.PERCENTAGE },
    layout: TableLayoutType.FIXED,
    borders: allNoBorders,
    rows: [new TableRow({
      height: { value: 16838, rule: "exact" },
      children: [new TableCell({
        shading: { type: ShadingType.CLEAR, fill: C.bg }, borders: noBorders,
        verticalAlign: "top",
        children,
      })],
    })],
  })];
}

// ── body helpers ──
function h1(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_1,
    keepNext: true,
    spacing: { before: 360, after: 160, line: 312 },
    children: [new TextRun({ text, bold: true, size: 32, color: P.primary, font: { ascii: "Calibri", eastAsia: "SimHei" } })],
  });
}
function h2(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_2,
    keepNext: true,
    spacing: { before: 240, after: 120, line: 312 },
    children: [new TextRun({ text, bold: true, size: 28, color: P.primary, font: { ascii: "Calibri", eastAsia: "SimHei" } })],
  });
}
function body(text, opts = {}) {
  return new Paragraph({
    alignment: AlignmentType.JUSTIFIED,
    indent: { firstLine: 480 },
    spacing: { line: 312, after: opts.after ?? 80 },
    children: (Array.isArray(text) ? text : [{ text }]).map(r => new TextRun({
      text: r.text, bold: r.bold, size: 24, color: r.color || P.body,
      font: { ascii: "Times New Roman", eastAsia: r.bold ? "SimHei" : "SimSun" },
    })),
  });
}
function mono(text) {
  return new Paragraph({
    spacing: { line: 276, after: 40 },
    indent: { left: 480 },
    shading: { type: ShadingType.CLEAR, fill: P.table.surface },
    children: [new TextRun({ text, size: 18, color: "1A3A4A", font: { ascii: "Courier New", eastAsia: "SimSun" } })],
  });
}

let figNo = 0, tabNo = 0;
function pngSize(p) { const b = fs.readFileSync(p); return { w: b.readUInt32BE(16), h: b.readUInt32BE(20) }; }
function figure(file, captionText, maxW = 586, maxH = 820) {
  figNo += 1;
  const full = path.join(ASSETS, file);
  const { w, h } = pngSize(full);
  const scale = Math.min(maxW / w, maxH / h, 1);
  const width = Math.round(w * scale), height = Math.round(h * scale);
  return [
    new Paragraph({
      alignment: AlignmentType.CENTER, keepNext: true, spacing: { before: 120, after: 40 },
      children: [new ImageRun({ data: fs.readFileSync(full), transformation: { width, height }, type: "png" })],
    }),
    new Paragraph({
      alignment: AlignmentType.CENTER, spacing: { after: 200 },
      children: [new TextRun({ text: `图${figNo}  ${captionText}`, size: 21, color: P.secondary,
        font: { ascii: "Times New Roman", eastAsia: "SimSun" } })],
    }),
  ];
}
function table(captionText, headers, rows, widths) {
  tabNo += 1;
  const cap = new Paragraph({
    keepNext: true, spacing: { before: 160, after: 60 },
    children: [new TextRun({ text: `表${tabNo}  ${captionText}`, bold: true, size: 21, color: P.primary,
      font: { ascii: "Times New Roman", eastAsia: "SimHei" } })],
  });
  const headerRow = new TableRow({
    tableHeader: true, cantSplit: true,
    children: headers.map((text, i) => new TableCell({
      shading: { type: ShadingType.CLEAR, fill: P.table.headerBg },
      margins: { top: 60, bottom: 60, left: 120, right: 120 },
      width: { size: widths[i], type: WidthType.PERCENTAGE },
      children: [new Paragraph({ spacing: { line: 276 },
        children: [new TextRun({ text, bold: true, size: 20, color: P.table.headerText,
          font: { ascii: "Calibri", eastAsia: "SimHei" } })] })],
    })),
  });
  const dataRows = rows.map((cells, ri) => new TableRow({
    cantSplit: true,
    children: cells.map((text, i) => new TableCell({
      shading: { type: ShadingType.CLEAR, fill: ri % 2 === 0 ? "FFFFFF" : P.table.surface },
      margins: { top: 50, bottom: 50, left: 120, right: 120 },
      width: { size: widths[i], type: WidthType.PERCENTAGE },
      children: [new Paragraph({ spacing: { line: 276 },
        children: [new TextRun({ text: String(text), size: 20, color: P.body,
          font: { ascii: "Times New Roman", eastAsia: "SimSun" } })] })],
    })),
  }));
  const tbl = new Table({
    width: { size: 100, type: WidthType.PERCENTAGE },
    borders: {
      top: { style: BorderStyle.SINGLE, size: 4, color: P.table.headerBg },
      bottom: { style: BorderStyle.SINGLE, size: 4, color: P.table.headerBg },
      left: NB, right: NB,
      insideHorizontal: { style: BorderStyle.SINGLE, size: 1, color: P.table.innerLine },
      insideVertical: NB,
    },
    rows: [headerRow, ...dataRows],
  });
  return [cap, tbl, new Paragraph({ spacing: { after: 120 }, children: [] })];
}

function pageNumFooter() {
  return new Footer({ children: [new Paragraph({
    alignment: AlignmentType.CENTER,
    children: [new TextRun({ children: [PageNumber.CURRENT], size: 18, color: "888888" })],
  })] });
}
function docHeader() {
  return new Header({ children: [new Paragraph({
    alignment: AlignmentType.RIGHT,
    border: { bottom: { style: BorderStyle.SINGLE, size: 2, color: P.table.innerLine, space: 4 } },
    children: [new TextRun({ text: "HIA 脸手数据集构建状态报告 · 2026-10-31", size: 16, color: "999999",
      font: { ascii: "Calibri", eastAsia: "SimSun" } })],
  })] });
}

// ══ body content ══
const bodyChildren = [];

// ── 1. 概览 ──
bodyChildren.push(h1("一、数据集概览与一页结论"));
bodyChildren.push(body("HIA 脸手数据集服务于 Qwen-Image-2.1 单人身体/面部/手部联合控制适配器(Native V5)的训练与评测。源数据为公开 HIA-RAW 数据集(25 个分片、25,000 段视频,约 112 GB),经脸手清晰度门禁、去手部重影门禁、跨视频 pHash 近重复检测、视频级 train/val/test 划分与有界动作配对后,形成带区域 loss_mask 与 effective_confidence 的原生分辨率 RGB + 134 点 2D pose 发布版;在此之上独立生成主体 SMPL-X 几何、度量深度、法线、mesh mask 与部位标签共九类产物。"));
bodyChildren.push(body([{ text: "截至 2026-10-31 的一页结论:", bold: true }]));
bodyChildren.push(body("1) 全量发布 HIA_processed_v2_full_20261027:7,167 个 main clips、22,276 帧、52,743 个训练 pairs(train 41,902 / val 5,526 / test 5,315);strict 子集 8,322 帧、17,995 pairs。相比 V1(96 帧 / 81 对),帧数提升 232 倍、pairs 提升 651 倍。"));
bodyChildren.push(body("2) 人体几何:冻结流水线(pipeline 3e21b2c1)完成 22,263 帧,九类产物全部通过 SHA256 复核;本次诊断并修复了全部 13 个拒检帧——修复 12 帧、排除 1 帧,合并覆盖达到 22,275 完成 + 1 排除 + 0 待处理,几何侧账目闭合。"));
bodyChildren.push(body("3) 根因发现:13 个拒检帧中的 6 帧属于误杀。上游 SMPLest-X(含官方 main/inference.py)将 load_img 返回的 RGB float32 数组直接交给 ultralytics YOLO,而 ultralytics 按 BGR 解释 numpy 输入并再次翻转通道,检测置信度被系统性削弱。该通道序问题在冻结流水线中如实复刻了官方行为,修复在独立签名副本中完成,冻结产物零改动。"));
bodyChildren.push(body("4) Native V5 typed cache(512 canvas 训练缓存)装配进行中:train pairs 已装配 33,650+/41,902,val/test 排队;32 个 train pairs 因 11 个几何失败帧暂缺,其中修复的 12 帧对应 pairs 可回填,被排除帧对应 pairs 永久剔除。全量装配与预检完成前不宣称训练就绪。"));
bodyChildren.push(body("5) 环境告警:节点 nvidia-smi 因 NVML 用户态库(580.178.04)与内核驱动(580.173.02)版本错配而不可用;CUDA 计算本身正常(torch CUDA tensor 运算通过),建议安排重启对齐。"));

// ── 2. 规模演进 ──
bodyChildren.push(h1("二、从 V1 到 V2 全量:规模演进与质量门禁"));
bodyChildren.push(h2("2.1 构建流水线概述"));
bodyChildren.push(body("V1(HIA_face_hand_v1_20261026)要求脸与双手同时 laplacian_var >= 60 并逐张人工质检,最终仅 96 帧 / 31 视频 / 81 对,规模成为瓶颈。V2 重构了门禁:SCRFD 五点人脸特征区清晰度 + 独立 MediaPipe 21 点手部确认 + 去手部重影多尺度频域门禁(gate hia_visible_v2_antighost_2:min_laplacian_var=32.0、min_gradient_mean=1.65、min_normalized_laplacian=0.090、min_hf_ratio=0.0080、min_multiscale_ratio=5.40、min_handedness_score=0.78),并以 64-bit DCT pHash 做跨视频近重复检测。出画手(out_of_frame)与躯干/衣物遮挡手(occluded)显式标记并在 loss_mask 中关闭该侧手部监督;可见糊手/重影手整帧拒绝。"));
bodyChildren.push(body("全量扫描 HIA_processed_v2_full_20261027 覆盖 25,000 段源视频记录,解码评估后得到 7,167 个 main clips、22,276 个 main 帧;按视频级划分(ratios 0.8/0.1/0.1,seed 20261026,同人物不同视频允许跨集合)与动作配对(min_time_gap 0.25s、每目标至多 4 个来源、min_pose_distance 0.06)生成训练 pairs,全部通过 verify_v2 全量哈希复核。"));
bodyChildren.push(h2("2.2 规模演进"));
bodyChildren.push(...table("HIA 脸手数据集规模演进(V1 → V2 全量)", ["版本", "视频/clips", "帧数", "训练 pairs", "strict 帧", "strict pairs"], [
  ["V1 (20261026)", "31", "96", "81", "—", "—"],
  ["V2 子集 release (1500 视频抽样)", "465", "1,320", "2,795", "534", "1,061"],
  ["V2 全量 release (7167 clips)", "7,167", "22,276", "52,743", "8,322", "17,995"],
  ["V2 全量·strict 子集单独计", "3,162", "8,322", "17,995", "—", "—"],
], [30, 14, 12, 16, 14, 14]));
bodyChildren.push(...figure("scale_evolution.png", "V1 → V2 全量规模演进(对数轴):视频数、帧数、训练 pairs"));
bodyChildren.push(h2("2.3 质检图册与人工抽检"));
bodyChildren.push(body("V1 的六帧原生 ROI 预览与 V2 的去重影盲审图册(qc/calibration_blind_v3,1,356 帧解码评估,拦截 619 帧模糊/重影手)构成质量门禁的人工证据链;16 个分层全部批准的审计规则冻结于 quality_gate_audit_v2.json。全量发布的每帧 RGB、关键点与 phash 均带 SHA256,可回验到 HIA-RAW 的 zip 分片。"));
bodyChildren.push(...figure("v1_preview.png", "V1 发布版六帧原生 ROI 预览(2026-10-26)", 420, 560));
bodyChildren.push(...figure("qc_sheet_047.png", "V2 去重影门禁盲审图册样例(sheet_047):左列 pass / 右列拦截", 480, 470));

// ── 3. 全量发布构成 ──
bodyChildren.push(h1("三、全量发布构成:划分、配对与区域监督"));
bodyChildren.push(h2("3.1 划分与配对"));
bodyChildren.push(...table("V2 全量发布 split 构成(主集 / strict 子集)", ["split", "clips", "帧数", "pairs(主)", "strict 帧", "strict pairs"], [
  ["train", "5,736", "17,728", "41,902", "6,681", "14,366"],
  ["val", "715", "2,261", "5,526", "833", "1,836"],
  ["test", "716", "2,287", "5,315", "808", "1,793"],
  ["合计", "7,167", "22,276", "52,743", "8,322", "17,995"],
], [14, 14, 16, 20, 17, 19]));
bodyChildren.push(h2("3.2 手部状态与区域监督语义"));
bodyChildren.push(body("每帧双手独立标注三态:visible_clear(参与监督,前提为通过清晰度门禁)、occluded(躯干/衣物遮挡,按规则关闭该侧监督)、out_of_frame(出画,同样关闭监督)。loss_mask 作用域为 regional_auxiliary_supervision:面部与躯干监督保持,出画/遮挡手不因几何重建而恢复监督,可见糊手整帧拒绝。"));
bodyChildren.push(...table("双手状态计数(共 22,276 帧)", ["状态", "hand_a", "hand_b"], [
  ["visible_clear(参与监督)", "10,793", "10,754"],
  ["occluded(遮挡,关闭该侧监督)", "3,520", "3,308"],
  ["out_of_frame(出画,关闭该侧监督)", "7,963", "8,214"],
], [46, 27, 27]));
bodyChildren.push(h2("3.3 训练样例展示"));
bodyChildren.push(body("showcase 从发布集中选取大动作幅度的 source-target 对,横幅展示 RGB、SMPL-X 3D 参数、134 点 3D DWPose、度量深度、法线、mesh mask 与部位标签;这些条件与 Native V5 typed cache 实际装配的内容一一对应。"));
bodyChildren.push(...figure("pair1_horizontal_strip.png", "showcase pair1 大动作对横幅(自左至右:source RGB/3D/深度/法线/mask/部位 → target 同序列)", 586, 170));
bodyChildren.push(...figure("pair2_horizontal_strip.png", "showcase pair2 大动作对横幅", 586, 170));
bodyChildren.push(...figure("hia_training_showcase_3pairs_large_motion.png", "三对大动作样本合并展示(2×3 面板)", 560, 480));

// ── 4. 几何层 ──
bodyChildren.push(h1("四、人体几何层:九类产物与可视化"));
bodyChildren.push(h2("4.1 产物构成与语义承诺"));
bodyChildren.push(body("几何补全按 2026-10-29 实施计划执行:保留发布版 RGB、2D pose、视频归属与质量门,独立生成原生画布主体几何;单目估计与显示归一化分开保存;失败帧不伪造条件入训。每帧九类产物全部落盘并记录 SHA256,geometry_coverage 复核 file_hashes_checked=true。"));
bodyChildren.push(...table("每帧几何产物(22,263 帧 × 9 类,全部哈希校验)", ["产物", "内容与语义"], [
  ["smplx_params.pt", "SMPL-X 参数(root/body/hand/jaw/expression/betas/mesh_cam/cam_trans)"],
  ["depth_metric.npy", "camera-space 顶点 Z,模型尺度,非场景量测真值;仅描述主体"],
  ["geometry.npz", "mesh_cam、137 关节、134 点 xyz/xy 重排、in-frame 可见性、focal/principal"],
  ["normal.png", "Trimesh 平滑视向法线(nvdiffrast 渲染,黑背景)"],
  ["depth_vis.png", "显示归一化深度图(前景 1st/99th 百分位,near=255)"],
  ["smplx_mask.png", "主体 mesh mask(黑背景)"],
  ["part_map.png", "55 组部位索引图(与 mask 逐像素一致性强校验)"],
  ["dwpose.png", "smplx137→dwpose134 重排投影骨架渲染"],
  ["camera.json", "内参、相机约定(+X右/+Y下/+Z远)、模型文件与 SHA、语义声明"],
], [26, 74]));
bodyChildren.push(h2("4.2 完成帧可视化"));
bodyChildren.push(body("以下六联图取自 showcase pair 的 source/target 帧(冻结流水线产物):RGB → 134 点 DWPose → 度量深度 → 法线 → SMPL-X mask → 部位标签(按索引着色)。身体拟合在舞蹈、蹲姿等大幅动作下与 2D 骨架保持一致。"));
bodyChildren.push(...figure("six_panel_1.png", "冻结流水线完成帧六联图(第 1 部分):RGB | DWPose | 深度 | 法线 | mask | 部位", 560, 540));
bodyChildren.push(...figure("six_panel_2.png", "冻结流水线完成帧六联图(第 2 部分)", 560, 540));
bodyChildren.push(...figure("geometry_smoke_6.png", "几何 smoke 阶段六帧产物(HIA_geometry_smoke_20261029,先行验证用)", 400, 780));

// ── 5. 拒检帧诊断与修复 ──
bodyChildren.push(h1("五、13 个拒检帧:诊断与修复(本次核心增量)"));
bodyChildren.push(h2("5.1 现象与账目"));
bodyChildren.push(body("全量几何作业(20261029T151254Z_df4ba96b)结束后:22,276 期望帧中 21,787 新完成 + 476 缓存命中 = 22,263 完成,13 帧拒检(11 train + 2 test),summary.complete=false。拒检原因全部为同一异常:smplestx_subject_disagrees_with_hia_primary——YOLOv8x 检出的最大人体框与发布版 primary 框 IoU < 0.3 时,流水线按冻结策略拒绝拟合,不伪造几何。"));
bodyChildren.push(h2("5.2 逐帧诊断"));
bodyChildren.push(body("对 13 帧逐一复检:以发布 manifest 的 primary 框(绿)、修正 BGR 输入后的 YOLOv8x 检出框(红/蓝)叠加渲染诊断图,并结合同 clip 兄弟帧完成度做目检判定。诊断证据与逐帧数字落在 repair_review_20261031/diagnosis.json 与 diag/*.png。"));
bodyChildren.push(...table("13 个拒检帧逐帧判定(BGR 修正输入下复检)", ["frame_id", "split", "检出数", "最大框IoU", "诊断结论", "处置"], [
  ["002830_000080", "train", "0", "—", "特殊服装,官方阈值无检出", "已修复·primary回退"],
  ["005692_000108", "train", "4", "0.152", "检测碎裂(大裙摆),最大框仅腿部", "已修复·判定primary"],
  ["005692_000122", "train", "4", "0.147", "检测碎裂(大裙摆)", "已修复·判定primary"],
  ["006303_000013", "train", "0", "—", "翻拍屏内容,无检出", "已修复·primary回退"],
  ["009295_000055", "test", "8", "0.780", "通道序退化误杀", "已修复·YOLO框"],
  ["009295_000254", "test", "4", "0.989", "通道序退化误杀", "已修复·YOLO框"],
  ["010239_000005", "train", "28", "0.895", "通道序退化误杀", "已修复·YOLO框"],
  ["011434_000187", "train", "8", "0.970", "通道序退化误杀", "已修复·YOLO框"],
  ["017323_000023", "train", "16", "0.181", "primary 框被 LED 大屏翻拍人脸污染", "排除并记账"],
  ["017323_000042", "train", "24", "0.872", "通道序退化误杀", "已修复·YOLO框"],
  ["023549_000187", "train", "8", "0.012", "台上主体未检出,检出为台下观众", "已修复·判定primary"],
  ["023549_000204", "train", "12", "0.014", "同上", "已修复·判定primary"],
  ["023549_000220", "train", "16", "0.003", "同上", "已修复·判定primary"],
], [19, 8, 8, 11, 32, 22]));
bodyChildren.push(...figure("005692_000108_diag.png", "诊断样例 1(检测碎裂):绿=发布 primary(正确),蓝=YOLO 最大框(仅腿部),IoU 0.152 被拒", 586, 300));
bodyChildren.push(...figure("023549_000187_diag.png", "诊断样例 2(主体漏检):台上红裙主体无检出,检出框全部落在台下观众头部", 586, 300));
bodyChildren.push(...figure("017323_000023_diag.png", "诊断样例 3(LED 屏污染):primary 框上探进大屏翻拍人脸,YOLO 框反而贴合台上真人;该帧排除", 586, 300));
bodyChildren.push(...figure("009295_000254_diag.png", "诊断样例 4(通道序误杀):BGR 修正输入下最大框与 primary IoU 0.989,本应直接通过门禁", 586, 300));
bodyChildren.push(h2("5.3 根因:YOLO 输入通道序"));
bodyChildren.push(body("冻结流水线与上游 SMPLest-X 官方 main/inference.py 完全一致地写了 detector.predict(load_img(path)):load_img 返回 RGB float32,而 ultralytics 对 numpy 输入按 BGR 约定再翻转一次通道,网络实际收到的是错误通道序的图像。人体检测对通道序大体鲁棒,因此 22,263 帧照常通过;但对边界样本是系统性削弱——6 帧在修正为 BGR uint8 输入后,最大框与 primary 的 IoU 恢复到 0.78 至 0.99,远超 0.3 门禁。该问题属上游怪癖,修复记录为修复副本的有意偏离,而非对冻结行为的事后改写。"));
bodyChildren.push(h2("5.4 修复方案与执行"));
bodyChildren.push(body("修复脚本为冻结脚本的副本(experiments/hia_geometry_repair_20261031/scripts/hia_geometry.py),仅含两处记录在案的偏离:其一,YOLO 输入改为 BGR uint8;其二,新增 --adjudicated-primary-frame-id 参数,对目检确认主体无歧义而检测碎裂/漏检的 5 帧,改用受质量门约束的发布 primary 框拟合,并在每条记录的 estimator.box_selection 中留痕。修复写入独立输出根 HIA_geometry_repair_v1_20261031,带自己的 pipeline_signature(sha256 前缀 df6f8f5c069ad027)、锁、frame index 与失败账本;冻结根与 24 个被签名的代码/模型文件零改动。12 帧全部一次成功,耗时 52 秒:5 帧官方 YOLO 框 + 2 帧 primary 回退 + 5 帧判定 primary。017323_000023 不予修复,以排除记账处理(其 primary 框本身已被 LED 屏内容污染,强行拟合反而引入错误几何)。"));
bodyChildren.push(...figure("repaired_frames_sheet_1.png", "修复帧产物对照(第 1 部分,6 帧):RGB | DWPose | 深度 | 法线,行首标注处置方式", 330, 820));
bodyChildren.push(...figure("repaired_frames_sheet_2.png", "修复帧产物对照(第 2 部分,6 帧)", 330, 820));
bodyChildren.push(h2("5.5 修复后账目"));
bodyChildren.push(...table("几何覆盖合并账目(merged_coverage_20261031.json)", ["账目项", "帧数", "说明"], [
  ["期望帧(release manifests)", "22,276", "train 17,728 / val 2,261 / test 2,287"],
  ["冻结流水线完成", "22,263", "pipeline 3e21b2c1,九类产物哈希复核通过"],
  ["修复副本补齐", "12", "签名 df6f8f5c,12/12 成功、0 失败"],
  ["排除并记账", "1", "017323_000023,LED 屏翻拍污染,证据 diag/017323_000023_diag.png"],
  ["待处理", "0", "22,263 + 12 + 1 = 22,276,账目闭合"],
], [34, 14, 52]));

// ── 6. native v5 ──
bodyChildren.push(h1("六、Native V5 训练接口与 typed cache 进展"));
bodyChildren.push(h2("6.1 装配状态"));
bodyChildren.push(body("hia_training.py prepare 正在把 release + geometry 装配为 Native V5 typed cache(输出根 HIA_native_v5_20261029,canvas 长边 512)。每帧目录存条件与元数据,每对目录存 body.npz(源/目标画布条件、区域有效性)、supervision.npz、local.npz 与 prompt_bank.json,内容寻址、原子落盘。截至 2026-10-31 00:45:train pairs 已装配 33,650+/41,902,val/test 排队;进程健康(约 289% CPU,已连续运行 22 小时)。日志中的 missing_geometry train=32 对应 11 个冻结失败帧所涉及的 pairs;其中 12 个已修复帧的 pairs 待全量装配完成后回填,被排除帧的 pairs 永久剔除并记账。"));
bodyChildren.push(h2("6.2 预检与测试基线"));
bodyChildren.push(body("2026-10-29 计划记录的基线仍然有效:三 split 各一对真实样本通过了 Native V5 数据读取与 consumer/loss 检查;本地契约测试 25 passed / 1 skipped(本地未同步远端 loss 模块);smoke 链 HIA_geometry_smoke_20261029(42 MB)与 HIA_native_v5_smoke_20261029(41 MB)先行验证了装配语义。几何与 Native V5 接口代码已实现,全量 typed cache 与全量 DataPlan 预检完成前,不宣称训练就绪。"));

// ── 7. 风险与下一步 ──
bodyChildren.push(h1("七、风险、告警与下一步"));
bodyChildren.push(...table("风险与待办清单", ["级别", "事项", "建议"], [
  ["告警", "NVML 版本错配:内核驱动 580.173.02 vs 用户态库 580.178.04,nvidia-smi 不可用", "CUDA 计算未受影响;安排节点重启对齐驱动,恢复 GPU 监控"],
  ["观察", "clip 017323 存在 LED 大屏翻拍内容,017323_000023 已排除;同 clip 其余帧(如 017323_000042)primary 与 YOLO 框 IoU 0.872 属正常", "建议 V3 发布前对疑似舞台屏 clip 做一次专项排查"],
  ["观察", "006303_000013 为翻拍屏帧,release 门禁已接受;修复按 primary 回退拟合屏中人", "保留现状;V3 可考虑在门禁中显式标记 recaptured-screen 分层"],
  ["待办", "12 个修复帧的 typed cache pairs 回填", "全量 prepare 完成后,以冻结根+修复根合并视图 resume 装配,并复核 pairs 计数"],
  ["待办", "全量 typed cache 装配与 DataPlan 预检(train 剩余约 8,252 对 + val/test 全量)", "等待当前作业完成后运行 verify 与预检,通过后再评估训练启动"],
  ["待办", "本地仓库与远端 experiments 目录结构漂移(远端仍是旧 5 族分组)", "按 experiments/README 约定在下次提交时统一整理"],
], [10, 48, 42]));
bodyChildren.push(body("总体判断:HIA 数据集的发布、几何与训练接口三层账目均已闭合或接近闭合——发布版 22,276 帧全部可追溯;几何覆盖 22,275 完成 + 1 排除;typed cache 装配过半且缺口来源明确。数据侧已具备进入全量预检的条件。"));

// ── 附录 ──
bodyChildren.push(h1("附录:关键路径与可复现命令"));
bodyChildren.push(body("远端数据根 sg:/data/shangguanrz/pic-edit-datasets/:HIA-RAW(源,112 GB)、HIA_processed_v2_full_20261027(全量发布)、HIA_geometry_v1_20261029(冻结几何,170 GB)、HIA_geometry_repair_v1_20261031(修复,本次新增)、HIA_native_v5_20261029(typed cache,116 GB)、repair_review_20261031(诊断与合并覆盖证据)。本地镜像实验目录 experiments/hia_geometry_repair_20261031/(脚本、诊断图、报告素材)。"));
bodyChildren.push(mono("# 13 帧拒检诊断(只读,输出到 repair_review_20261031)"));
bodyChildren.push(mono("python hia_failure_diagnosis.py --release <full_release> --geometry-root <HIA_geometry_v1_20261029> --output <repair_review_20261031>"));
bodyChildren.push(mono("# 12 帧修复(独立签名副本,冻结根零改动;52 秒完成)"));
bodyChildren.push(mono("python hia_geometry.py --fallback-hia-primary --frame-id <id x12> --adjudicated-primary-frame-id <id x5> --output <HIA_geometry_repair_v1_20261031>"));
bodyChildren.push(mono("# 合并覆盖账目与报告素材"));
bodyChildren.push(mono("python build_report_assets.py  # merged_coverage_20261031.json + 六联图/修复对照/规模图"));
bodyChildren.push(body("诊断结论的原始数字见 repair_review_20261031/diagnosis.json(每帧检出框、IoU、primary 覆盖率、兄弟帧完成度);修复记录见 HIA_geometry_repair_v1_20261031/frame_index_*.jsonl(每帧 estimator.box_selection 与九类产物哈希)。", { after: 200 }));

// ══ document assembly ══
const pgSize = { width: 11906, height: 16838 };
const pgMargin = { top: 1440, bottom: 1440, left: 1701, right: 1417 };

const doc = new Document({
  styles: {
    default: {
      document: {
        run: { font: { ascii: "Times New Roman", eastAsia: "SimSun" }, size: 24, color: "000000" },
        paragraph: { spacing: { line: 312 } },
      },
      heading1: {
        run: { font: { ascii: "Calibri", eastAsia: "SimHei" }, size: 32, bold: true, color: P.primary },
        paragraph: { spacing: { before: 360, after: 160, line: 312 }, outlineLevel: 0 },
      },
      heading2: {
        run: { font: { ascii: "Calibri", eastAsia: "SimHei" }, size: 28, bold: true, color: P.primary },
        paragraph: { spacing: { before: 240, after: 120, line: 312 }, outlineLevel: 1 },
      },
    },
  },
  sections: [
    { // Section 1: cover — margin 0, no footer, no pageNumbers
      properties: { page: { size: pgSize, margin: { top: 0, bottom: 0, left: 0, right: 0 } } },
      children: buildCoverR1({
        palette: P,
        englishLabel: "HIA FACE-HAND DATASET STATUS",
        title: "HIA 脸手数据集构建状态报告",
        subtitle: "规模演进 · 人体几何补全 · 拒检帧诊断修复 · 训练缓存进展",
        metaLines: [
          "报告日期:2026-10-31",
          "数据根目录:sg:/data/shangguanrz/pic-edit-datasets/",
          "全量发布:HIA_processed_v2_full_20261027(22,276 帧 / 52,743 pairs)",
          "几何账目:22,275 完成 + 1 排除 + 0 待处理(修复签名 df6f8f5c)",
        ],
        footerLeft: "pic-edit · Qwen-Image-2.1 姿态控制数据准备",
        footerRight: "2026-10-31",
      }),
    },
    { // Section 2: front matter (abstract + TOC) — Roman numerals
      properties: {
        type: SectionType.NEXT_PAGE,
        page: { size: pgSize, margin: pgMargin, pageNumbers: { start: 1, formatType: NumberFormat.UPPER_ROMAN } },
      },
      footers: { default: pageNumFooter() },
      children: [
        new Paragraph({
          alignment: AlignmentType.CENTER, spacing: { before: 480, after: 360 },
          children: [new TextRun({ text: "摘  要", bold: true, size: 32, font: { eastAsia: "SimHei", ascii: "Times New Roman" }, color: P.primary })],
        }),
        body("本报告盘点 HIA 脸手数据集的构建现状:全量发布覆盖 7,167 个 main clips、22,276 帧、52,743 个训练 pairs,strict 子集 8,322 帧、17,995 pairs;人体几何冻结流水线完成 22,263 帧九类产物并通过全量哈希复核。本次工作诊断了全部 13 个几何拒检帧,确认其中 6 帧源于上游 SMPLest-X 向 ultralytics YOLO 传递 RGB 数组导致的通道序退化误杀,并以独立签名副本修复 12 帧、排除 1 帧(LED 大屏翻拍污染),使几何账目闭合至 22,275 完成 + 1 排除 + 0 待处理。Native V5 typed cache 装配进行中(train 33,650+/41,902 pairs)。全文含 12 组可视化:规模演进、质检图册、showcase 大动作对、几何六联图、逐帧诊断与修复对照。"),
        new Paragraph({ spacing: { after: 200 }, children: [] }),
        new Paragraph({
          alignment: AlignmentType.CENTER, spacing: { before: 200, after: 360 },
          children: [new TextRun({ text: "目  录", bold: true, size: 32, font: { eastAsia: "SimHei", ascii: "Times New Roman" }, color: P.primary })],
        }),
        new TableOfContents("Table of Contents", { hyperlink: true, headingStyleRange: "1-2" }),
        new Paragraph({
          spacing: { before: 200 },
          children: [new TextRun({
            text: "注:本目录由域代码生成。编辑后如需刷新页码,请在目录上点击右键并选择「更新域」。",
            italics: true, size: 18, color: "888888", font: { eastAsia: "SimSun", ascii: "Times New Roman" } })],
        }),
        // No trailing PageBreak here: the body section below is SectionType.NEXT_PAGE,
        // and PageBreak + NEXT_PAGE would produce a blank page (postcheck rule).
      ],
    },
    { // Section 3: body — Arabic from 1
      properties: {
        type: SectionType.NEXT_PAGE,
        page: { size: pgSize, margin: pgMargin, pageNumbers: { start: 1, formatType: NumberFormat.DECIMAL } },
      },
      headers: { default: docHeader() },
      footers: { default: pageNumFooter() },
      children: bodyChildren,
    },
  ],
});

Packer.toBuffer(doc).then(buf => {
  fs.writeFileSync(OUT, buf);
  console.log("WROTE", OUT, buf.length, "bytes; figures:", figNo, "tables:", tabNo);
});
