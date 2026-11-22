# -*- coding: utf-8 -*-
"""Scrapling 冒烟测试：抓 B 站搜索页视频卡片的 BV/标题/播放量。单页快速验证解析选择器。"""
from scrapling.fetchers import StealthyFetcher

URL = "https://search.bilibili.com/all?keyword=%E5%8F%8C%E4%BA%BA%E8%88%9E&order=click"

page = StealthyFetcher.fetch(URL, headless=True, network_idle=True, timeout=60_000)


def first(el, sel):
    r = el.css(sel)
    return r[0] if r else None


cards = page.css("div.bili-video-card")
print("CARDS", len(cards))
for card in cards[:10]:
    a = first(card, ".bili-video-card__info--tit a") or first(card, 'a[href*="/video/BV"]')
    if a is None:
        continue
    href = a.attrib.get("href", "")
    bv = href.split("/video/")[-1].split("?")[0].split("/")[0]
    title = (a.attrib.get("title") or a.text or "").strip()
    meta = first(card, ".bili-video-card__details span:nth-child(1)")
    plays = (meta.text or "").strip() if meta else ""
    print(bv, "|", plays, "|", title[:50])
