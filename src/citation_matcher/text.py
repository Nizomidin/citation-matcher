from __future__ import annotations

import re
from html import unescape

_TAG_RE = re.compile(r"<[^>]+>")


def clean_html(text: str | None) -> str:
    """Remove HTML tags and decode entities like &amp; → &."""
    if not text:
        return ""
    text = unescape(str(text))
    text = _TAG_RE.sub("", text)
    return re.sub(r"\s+", " ", text).strip()
