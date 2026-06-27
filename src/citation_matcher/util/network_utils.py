from __future__ import annotations

import json
import logging
import os
from typing import Any
from uuid import uuid4

import requests
from curl_cffi.requests import AsyncSession

from citation_matcher.config import (
    ELIBRARY_COOKIE_FILE,
    ELIBRARY_HEADERS,
    ELIBRARY_HOME_URL,
    REQUEST_TIMEOUT,
)

logger = logging.getLogger(__name__)


def get_proxy() -> str | None:
    password = os.getenv("PROXY_PASSWORD")
    server = os.getenv("PROXY_SERVER")
    username = os.getenv("PROXY_USERNAME")
    if not all((password, server, username)):
        return None
    session_id = uuid4().hex
    return f"http://{username}-country-ru-session-{session_id}:{password}@{server}"


def load_cookies() -> dict[str, str]:
    if not ELIBRARY_COOKIE_FILE.exists():
        return {}
    with ELIBRARY_COOKIE_FILE.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return data if isinstance(data, dict) else {}


def save_cookies(cookies: dict[str, str]) -> None:
    ELIBRARY_COOKIE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with ELIBRARY_COOKIE_FILE.open("w", encoding="utf-8") as handle:
        json.dump(cookies, handle)


async def start_session(proxy: str | None = None) -> dict[str, str] | None:
    async with AsyncSession(
        impersonate="chrome", headers=ELIBRARY_HEADERS, proxy=proxy
    ) as session:
        response = await session.get(ELIBRARY_HOME_URL)
        if "page_captcha.asp" in str(response.url):
            logger.warning("eLibrary captcha during session start")
            return None
        return session.cookies.get_dict()


async def get_new_cookies(proxy: str | None = None) -> dict[str, str] | None:
    cookies = await start_session(proxy=proxy)
    if cookies:
        save_cookies(cookies)
        logger.info("eLibrary cookies saved to %s", ELIBRARY_COOKIE_FILE)
    return cookies


def request_json(method: str, url: str, **kwargs: Any) -> dict[str, Any] | None:
    try:
        response = requests.request(method, url, timeout=REQUEST_TIMEOUT, **kwargs)
    except requests.RequestException as exc:
        logger.debug("HTTP request failed: %s %s — %s", method, url, exc)
        return None
    if response.status_code != 200:
        return None
    try:
        return response.json()
    except ValueError:
        return None
