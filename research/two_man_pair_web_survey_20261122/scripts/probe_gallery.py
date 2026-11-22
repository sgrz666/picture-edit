# -*- coding: utf-8 -*-
"""Scrapling 探针：婚纱旅拍画廊站结构验证（纪梵希全球旅拍）。

验证 /work/list 列表页普通 HTTP 可抓、分类与套系链接结构，
为后续"按套系抓整组同 couple 多 pose 客片"铺路。输出仅打印，不落产物。
"""
from scrapling.fetchers import Fetcher, StealthyFetcher

LIST_URL = "https://www.jifanxilvpai.com/work/list"

try:
    page = Fetcher.get(LIST_URL, timeout=30_000)
    kind = "http"
except Exception as e:
    print("HTTP_FAIL", type(e).__name__, str(e)[:120], flush=True)
    page = StealthyFetcher.fetch(LIST_URL, headless=True, network_idle=True, timeout=60_000)
    kind = "stealth"

print("VIA", kind)


def first(el, sel):
    r = el.css(sel)
    return r[0] if r else None


t = first(page, "title")
print("TITLE", (t.text or "").strip()[:60] if t else "")

srcs = []
for im in page.css("img"):
    s = (
        im.attrib.get("src")
        or im.attrib.get("data-src")
        or im.attrib.get("data-original")
        or ""
    )
    if s.startswith("http") and any(
        k in s for k in ("thumb", "work", "jpg", "png", "webp", "image")
    ):
        srcs.append(s)
print("IMG_COUNT", len(page.css("img")), "CAND", len(srcs))
for s in srcs[:8]:
    print("IMG", s[:150])

works = []
for a in page.css("a"):
    href = a.attrib.get("href", "")
    text = (a.text or "").strip()
    if any(k in href for k in ("work", "detail", "show")) and href not in works:
        works.append(href)
        if len(works) <= 10:
            print("LINK", text[:30], "|", href[:120])
print("LINK_COUNT", len(works))
