# -*- coding: utf-8 -*-
"""B 站双人舞候选视频采集：多关键词搜索页 -> BV+标题+播放量，去重落 JSON。

Scrapling StealthyFetcher 打开 B 站搜索页（按播放量排序），解析视频卡片，
产出候选清单供既有 bilibili 视频管线（下载 -> 抽帧 -> two_man_video_pair_mine）消费。
用法：python harvest_bili.py，输出与脚本同目录 bilibili_duo_candidates.json。
"""
import json
import time

from scrapling.fetchers import StealthyFetcher

KEYWORDS = [
    "双人舞",
    "情侣双人舞",
    "双人翻跳",
    "姐妹双人舞",
    "男女双人舞",
    "双人舞蹈",
    "情侣 翻跳",
    "双人 cover 舞蹈",
]

seen = {}
for kw in KEYWORDS:
    url = "https://search.bilibili.com/all?keyword=" + kw.replace(" ", "%20") + "&order=click"
    try:
        page = StealthyFetcher.fetch(url, headless=True, network_idle=True, timeout=60_000)
    except Exception as e:
        print("FETCH_FAIL", kw, type(e).__name__, e, flush=True)
        continue
    cards = page.css("div.bili-video-card")
    got = 0

    def first(el, sel):
        r = el.css(sel)
        return r[0] if r else None

    for card in cards:
        a = first(card, ".bili-video-card__info--tit a") or first(card, 'a[href*="/video/BV"]')
        if a is None:
            continue
        href = a.attrib.get("href", "")
        if "BV" not in href:
            continue
        bv = href.split("/video/")[-1].split("?")[0].split("/")[0]
        title = (a.attrib.get("title") or a.text or "").strip()
        meta = first(card, ".bili-video-card__details span:nth-child(1)")
        plays = (meta.text or "").strip() if meta else ""
        up = first(card, ".bili-video-card__info--author")
        author = (up.text or "").strip() if up else ""
        if bv not in seen:
            seen[bv] = {"bv": bv, "title": title, "plays": plays, "author": author, "kw": kw}
            got += 1
    print("OK", kw, "cards=", len(cards), "new=", got, flush=True)
    time.sleep(3)

with open("bilibili_duo_candidates.json", "w", encoding="utf-8") as f:
    json.dump(list(seen.values()), f, ensure_ascii=False, indent=1)
print("TOTAL", len(seen))
