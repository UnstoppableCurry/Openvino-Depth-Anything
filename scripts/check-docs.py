#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Verify the docs/ static showcase: files, landmarks, links, sitemap, robots."""

from __future__ import annotations

import re
import sys
import xml.etree.ElementTree as ET
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlparse, unquote

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
SITE = "https://unstoppablecurry.github.io/Openvino-Depth-Anything"
REQUIRED_FILES = (
    "index.html",
    "architecture.html",
    "usage.html",
    "limitations.html",
    "404.html",
    "robots.txt",
    "sitemap.xml",
    "favicon.svg",
    "favicon.ico",
    "assets/site.css",
)
LANDMARKS = ("header", "nav", "main", "footer")
SITEMAP_NS = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}


class Page(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.tags: set[str] = set()
        self.ids: set[str] = set()
        self.refs: list[tuple[str, str]] = []
        self.lang = ""
        self._in_title = False
        self.title = ""

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.tags.add(tag)
        ad = {k: (v or "") for k, v in attrs}
        if tag == "html":
            self.lang = ad.get("lang", "")
        if "id" in ad:
            self.ids.add(ad["id"])
        if tag == "title":
            self._in_title = True
        for attr in ("href", "src"):
            if attr in ad:
                self.refs.append((attr, ad[attr]))

    def handle_endtag(self, tag: str) -> None:
        if tag == "title":
            self._in_title = False

    def handle_data(self, data: str) -> None:
        if self._in_title:
            self.title += data


def fail(errors: list[str]) -> None:
    for item in errors:
        print(f"FAIL: {item}", file=sys.stderr)
    raise SystemExit(1)


def resolve_local(page: Path, raw: str) -> Path | None:
    parsed = urlparse(raw)
    if parsed.scheme in {"http", "https", "mailto", "data"}:
        return None
    if raw.startswith("#"):
        return page
    if parsed.netloc:
        return None
    path = unquote(parsed.path)
    if not path:
        return page
    if path.startswith("/"):
        raise ValueError(f"absolute-root path {raw!r} breaks project Pages subpath")
    return (page.parent / path).resolve()


def main() -> None:
    errors: list[str] = []
    if not DOCS.is_dir():
        fail([f"missing directory {DOCS}"])

    for rel in REQUIRED_FILES:
        if not (DOCS / rel).is_file():
            errors.append(f"missing {rel}")

    html_files = sorted(DOCS.glob("*.html"))
    if not html_files:
        errors.append("no HTML files in docs/")

    for html_path in html_files:
        text = html_path.read_text(encoding="utf-8")
        page = Page()
        try:
            page.feed(text)
            page.close()
        except Exception as exc:  # parse problems are reported as check failures
            errors.append(f"{html_path.name}: HTML parse error: {exc}")
            continue

        rel = html_path.name
        if not page.lang.startswith("zh"):
            errors.append(f"{rel}: html lang should be Chinese-primary, got {page.lang!r}")
        if "content" not in page.ids:
            errors.append(f"{rel}: missing id='content' (skip-link target)")
        if not page.title.strip():
            errors.append(f"{rel}: missing <title>")
        for tag in LANDMARKS:
            if tag not in page.tags:
                errors.append(f"{rel}: missing landmark <{tag}>")

        for attr, raw in page.refs:
            if not raw:
                errors.append(f"{rel}: empty {attr}")
                continue
            try:
                target = resolve_local(html_path, raw)
            except ValueError as exc:
                errors.append(f"{rel}: {exc}")
                continue
            if target is None:
                continue
            if not str(target).startswith(str(DOCS.resolve())):
                errors.append(f"{rel}: {attr}={raw!r} escapes docs/")
                continue
            parsed = urlparse(raw)
            if parsed.path and not target.is_file():
                errors.append(f"{rel}: broken {attr} {raw!r}")
            if parsed.fragment and parsed.fragment not in Page_ids_or_self(page, html_path, target):
                # fragment on same file already parsed; other file checked loosely
                if target == html_path.resolve() and parsed.fragment not in page.ids:
                    errors.append(f"{rel}: missing fragment #{parsed.fragment}")

    robots = DOCS / "robots.txt"
    if robots.is_file():
        body = robots.read_text(encoding="utf-8")
        expected = f"Sitemap: {SITE}/sitemap.xml"
        if "User-agent:" not in body:
            errors.append("robots.txt: missing User-agent")
        if expected not in body:
            errors.append(f"robots.txt: missing `{expected}`")

    sitemap = DOCS / "sitemap.xml"
    if sitemap.is_file():
        try:
            tree = ET.parse(sitemap)
            locs = [el.text or "" for el in tree.findall(".//sm:loc", SITEMAP_NS)]
            if not locs:
                locs = [el.text or "" for el in tree.findall(".//{http://www.sitemaps.org/schemas/sitemap/0.9}loc")]
            if not locs:
                errors.append("sitemap.xml: no <loc> entries")
            expected_locs = {
                f"{SITE}/",
                f"{SITE}/architecture.html",
                f"{SITE}/usage.html",
                f"{SITE}/limitations.html",
            }
            missing = expected_locs.difference(locs)
            extra = set(locs).difference(expected_locs)
            if missing:
                errors.append(f"sitemap.xml: missing {sorted(missing)}")
            if extra:
                errors.append(f"sitemap.xml: unexpected {sorted(extra)}")
            for loc in locs:
                if not loc.startswith(SITE):
                    errors.append(f"sitemap.xml: loc outside intended host {loc!r}")
                tail = loc[len(SITE) :]
                if tail in {"", "/"}:
                    page_path = DOCS / "index.html"
                else:
                    page_path = DOCS / tail.lstrip("/")
                if not page_path.is_file():
                    errors.append(f"sitemap.xml: {loc} has no file {page_path.name}")
        except ET.ParseError as exc:
            errors.append(f"sitemap.xml: {exc}")

    if errors:
        fail(errors)
    print("docs check passed:")
    print(f"  pages: {', '.join(p.name for p in html_files)}")
    print(f"  landmarks: {', '.join(LANDMARKS)}")
    print(f"  sitemap/robots base: {SITE}")


def Page_ids_or_self(page: Page, html_path: Path, target: Path) -> set[str]:
    if target.resolve() == html_path.resolve():
        return page.ids
    return set()


if __name__ == "__main__":
    main()
