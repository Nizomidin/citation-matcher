from __future__ import annotations

import asyncio

from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse

from citation_matcher.matcher import match_citation

app = FastAPI(title="Citation Matcher", version="0.1.0")

_HTML = """<!DOCTYPE html>
<html lang="ru">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Citation Matcher</title>
  <style>
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
    body { font-family: 'Segoe UI', system-ui, sans-serif; background: #f0f4f8; color: #1a202c;
      min-height: 100vh; display: flex; flex-direction: column; align-items: center; padding: 40px 16px 60px; }
    header { text-align: center; margin-bottom: 36px; }
    header h1 { font-size: 2rem; font-weight: 700; color: #2b6cb0; }
    header p { margin-top: 6px; color: #4a5568; font-size: 0.95rem; }
    .card { background: #fff; border-radius: 12px; box-shadow: 0 4px 24px rgba(0,0,0,.08);
      padding: 32px; width: 100%; max-width: 760px; }
    .search-row { display: flex; gap: 10px; }
    .search-row textarea { flex: 1; resize: vertical; min-height: 90px; padding: 12px 14px;
      font-size: 0.95rem; border: 1.5px solid #cbd5e0; border-radius: 8px; outline: none;
      font-family: inherit; line-height: 1.5; }
    .search-row textarea:focus { border-color: #3182ce; }
    button[type=submit] { align-self: flex-end; padding: 12px 28px; background: #3182ce; color: #fff;
      border: none; border-radius: 8px; font-size: 1rem; font-weight: 600; cursor: pointer; }
    button[type=submit]:hover { background: #2c5282; }
    .spinner { display: none; margin: 32px auto; width: 40px; height: 40px; border: 4px solid #bee3f8;
      border-top-color: #3182ce; border-radius: 50%; animation: spin .7s linear infinite; }
    @keyframes spin { to { transform: rotate(360deg); } }
    .results { margin-top: 32px; }
    .error-box { background: #fff5f5; border: 1px solid #fed7d7; border-radius: 8px; padding: 16px; color: #c53030; }
    .best-match { background: #ebf8ff; border: 1.5px solid #90cdf4; border-radius: 10px; padding: 20px 24px; margin-bottom: 24px; }
    .best-match .badge { display: inline-block; background: #3182ce; color: #fff; font-size: 0.75rem;
      font-weight: 600; border-radius: 20px; padding: 2px 10px; margin-bottom: 10px; }
    .best-match .prob { font-size: 1.5rem; font-weight: 700; color: #2b6cb0; margin-bottom: 4px; }
    .best-match .prob-hint { font-size: 0.78rem; color: #718096; margin-bottom: 10px; }
    .best-match .citation-text { font-size: 0.95rem; color: #2d3748; margin-bottom: 12px; line-height: 1.6; }
    .best-match .meta { font-size: 0.82rem; color: #718096; }
    .best-match .meta a { color: #3182ce; text-decoration: none; }
    .source-tag { display: inline-block; font-size: 0.72rem; font-weight: 600; border-radius: 4px;
      padding: 1px 7px; margin-left: 6px; }
    .source-crossref { background: #fefcbf; color: #744210; }
    .source-openalex { background: #e9d8fd; color: #44337a; }
    .source-cyberleninka { background: #c6f6d5; color: #22543d; }
    .source-elibrary { background: #feebc8; color: #7b341e; }
    h3 { font-size: 1rem; font-weight: 600; color: #4a5568; margin-bottom: 12px; }
    .alt-list { list-style: none; display: flex; flex-direction: column; gap: 10px; }
    .alt-item { background: #f7fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 14px 16px;
      display: flex; align-items: flex-start; gap: 14px; }
    .alt-prob { font-size: 0.9rem; font-weight: 700; color: #3182ce; min-width: 52px; text-align: right; }
    .alt-title { font-size: 0.9rem; font-weight: 500; color: #2d3748; }
    .alt-meta { font-size: 0.78rem; color: #718096; margin-top: 3px; }
    footer { margin-top: 48px; font-size: 0.8rem; color: #a0aec0; text-align: center; }
  </style>
</head>
<body>
  <header>
    <h1>📖 Citation Matcher</h1>
    <p>Поиск по Crossref, OpenAlex, КиберЛенинке и eLibrary.</p>
  </header>
  <div class="card">
    <form id="searchForm" method="post" action="/match">
      <div class="search-row">
        <textarea name="query" id="queryInput" placeholder="Например: Deep learning LeCun Nature 2015">{{ query }}</textarea>
        <button type="submit">Найти</button>
      </div>
    </form>
    <div class="spinner" id="spinner"></div>
    {% if error %}<div class="results"><div class="error-box">⚠ {{ error }}</div></div>{% endif %}
    {% if best %}
    <div class="results">
      <div class="best-match">
        <span class="badge">Лучшее совпадение</span>
        <span class="source-tag source-{{ best.source }}">{{ best.source }}</span>
        <div class="prob">{{ "%.1f"|format(best.probability * 100) }}%</div>
        <div class="prob-hint">вероятность совпадения</div>
        <div class="citation-text">{{ best.formatted_citation }}</div>
        <div class="meta">
          {% if best.doi %}DOI: <a href="https://doi.org/{{ best.doi }}" target="_blank">{{ best.doi }}</a>{% endif %}
          {% if best.link and not best.doi and best.source == 'cyberleninka' %}<a href="https://cyberleninka.ru{{ best.link }}" target="_blank">КиберЛенинка</a>{% endif %}
          {% if best.link and not best.doi and best.source == 'elibrary' %}<a href="https://elibrary.ru{{ best.link }}" target="_blank">eLibrary</a>{% endif %}
          {% if best.journal %} · {{ best.journal }}{% endif %}
          {% if best.year %} · {{ best.year }}{% endif %}
        </div>
      </div>
      {% if alternatives %}
      <h3>Другие кандидаты</h3>
      <ul class="alt-list">
        {% for alt in alternatives %}
        <li class="alt-item">
          <div class="alt-prob">{{ "%.1f"|format(alt.probability * 100) }}%</div>
          <div>
            <div class="alt-title">{{ alt.title }} <span class="source-tag source-{{ alt.source }}">{{ alt.source }}</span></div>
            <div class="alt-meta">{% if alt.authors %}{{ alt.authors }}{% endif %}{% if alt.year %} · {{ alt.year }}{% endif %}</div>
          </div>
        </li>
        {% endfor %}
      </ul>
      {% endif %}
    </div>
    {% endif %}
  </div>
  <footer>Crossref · OpenAlex · КиберЛенинка · eLibrary</footer>
  <script>
    document.getElementById('searchForm').addEventListener('submit', function() {
      document.getElementById('spinner').style.display = 'block';
    });
  </script>
</body>
</html>"""


def _render(query="", best=None, alternatives=None, error=None) -> HTMLResponse:
    from jinja2 import Environment

    env = Environment(autoescape=True)
    return HTMLResponse(
        content=env.from_string(_HTML).render(
            query=query,
            best=best,
            alternatives=alternatives or [],
            error=error,
        )
    )


@app.get("/", response_class=HTMLResponse)
async def index() -> HTMLResponse:
    return _render()


@app.post("/match", response_class=HTMLResponse)
async def match(query: str = Form(...)) -> HTMLResponse:
    result = await asyncio.to_thread(match_citation, query)
    if result.error:
        return _render(query=query, error=result.error)
    return _render(query=query, best=result.best_match, alternatives=result.alternatives)


@app.post("/api/match")
async def api_match(request: Request) -> dict:
    body = await request.json()
    query = (body.get("query") or "").strip()
    if not query:
        return {"error": "Empty query"}
    result = await asyncio.to_thread(match_citation, query)
    return result.to_dict()
