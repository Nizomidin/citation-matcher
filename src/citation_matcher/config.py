from __future__ import annotations

import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
SEEDS_DIR = DATA_DIR / "seeds"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
REPORTS_DIR = DATA_DIR / "reports"
FIXTURES_DIR = PROJECT_ROOT / "tests" / "fixtures"
DEFAULT_ELIBRARY_SEEDS_CSV = SEEDS_DIR / "elibrary_data_fixed.csv"
MODELS_DIR = PROJECT_ROOT / "models"
DEFAULT_MODEL_PATH = MODELS_DIR / "model.pkl"
ELIBRARY_COOKIE_FILE = DATA_DIR / "elibrary_cookies.json"

# Runtime multi-search: eLibrary excluded (403 without RU proxy, captcha/redirect loops).
DEFAULT_MULTI_SEARCH_SOURCES = ("crossref", "openalex", "cyberleninka")

CROSSREF_API = "https://api.crossref.org/works"
OPENALEX_API = "https://api.openalex.org/works"
CYBERLENINKA_API = "https://cyberleninka.ru/api/search"
ELIBRARY_BASE = "https://www.elibrary.ru"
ELIBRARY_HOME_URL = f"{ELIBRARY_BASE}/defaultx.asp"
ELIBRARY_SEARCH_URL = f"{ELIBRARY_BASE}/query_results.asp"

USER_AGENT = "citation-matcher/0.1 (mailto:citation-matcher@example.com)"
DEFAULT_ROWS = 100
REQUEST_TIMEOUT = 20
DATASET_SEED = 42

_TAG_RE = re.compile(r"<[^>]+>")

ELIBRARY_HEADERS = {
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "ru,en;q=0.9",
    "Content-Type": "application/x-www-form-urlencoded",
    "Origin": ELIBRARY_BASE,
    "Referer": ELIBRARY_HOME_URL,
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/146.0.0.0 Safari/537.36"
    ),
}

ELIBRARY_SEED_QUERIES = [
    "экономика",
    "образование",
    "медицина",
    "информатика",
    "управление",
    "исследование",
    "анализ",
    "технологии",
    "машинное обучение",
    "право",
]
