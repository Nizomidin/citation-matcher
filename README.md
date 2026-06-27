# Citation Matcher

Программный инструмент для исправления ошибок в библиографических описаниях (тема 8).

Для заданного библиографического описания инструмент обращается к открытым наукометрическим базам — [Crossref](https://www.crossref.org/), [OpenAlex](https://openalex.org/), [КиберЛенинка](https://cyberleninka.ru/) и [eLibrary.ru](https://elibrary.ru/) — находит кандидатов, ранжирует их ML-моделью и возвращает наиболее вероятную статью с **вероятностью совпадения** и оформленной библиографической ссылкой.

## Установка

```bash
python -m venv venv
source venv/bin/activate
pip install -e .
```

## Подготовка данных и обучение

```bash
# Сбор датасета: 50 Crossref + 50 КиберЛенинка (eLibrary опционально)
python scripts/build_dataset.py

# С eLibrary (нужны cookies и часто прокси)
export $(cat .env | xargs)   # PROXY_USERNAME, PROXY_PASSWORD, PROXY_SERVER
python scripts/build_dataset.py --refresh-elibrary-cookies
python scripts/build_dataset.py --crossref 50 --cyberleninka 50 --elibrary 25

# Обучение калиброванной модели
python -m citation_matcher.train
```

Модель сохраняется в `models/model.pkl`.

## eLibrary.ru

eLibrary требует cookies и часто блокирует запросы без прокси. Скопируйте `.env.example` в `.env` и задайте переменные прокси. Cookies сохраняются в `data/elibrary_cookies.json`.

При captcha eLibrary пропускается, поиск продолжается по остальным базам.

## Использование

### Веб-интерфейс

```bash
source venv/bin/activate
uvicorn citation_matcher.app:app --reload
```

Откройте [http://localhost:8000](http://localhost:8000).

### CLI

```bash
python -m citation_matcher "ЭВОЛЮЦИЯ ТЕОРИИ ИНВЕСТИЦИЙ: ОТ ИСТОКОВ ДО СОВРЕМЕННОЙ ТЕОРИИ"
python -m citation_matcher --json "Deep learning LeCun Nature 2015"
```

### Python API

```python
from citation_matcher import match_citation

result = match_citation("Deep learning LeCun Nature 2015")
print(result.best_match["probability"])
print(result.best_match["formatted_citation"])
```

## Как это работает

1. **Поиск** — параллельные запросы в Crossref, OpenAlex и КиберЛенинку (eLibrary отключён в runtime: 403/captcha без RU-прокси).
2. **Признаки** — сходство заголовка, автора, ранг внутри базы, год, источник (one-hot).
3. **Ранжирование** — калиброванная LogisticRegression (`CalibratedClassifierCV`).
4. **Вероятность** — калиброванный `predict_proba`; для exact match по заголовку — boost.
5. **Результат** — лучший кандидат с оформленной ссылкой.

## Структура

```
citation-matcher/
├── src/citation_matcher/
│   ├── config.py          — пути, константы, API URLs
│   ├── search.py          — поиск по всем базам
│   ├── elibrary.py        — клиент eLibrary.ru
│   ├── matcher.py         — признаки, ранжирование, match_citation()
│   ├── dataset.py         — сбор обучающего датасета
│   ├── train.py           — обучение модели
│   ├── app.py             — веб-интерфейс (FastAPI)
│   ├── benchmark/         — офлайн-оценка и бенчмарки
│   │   ├── evaluate.py
│   │   ├── rankers.py
│   │   └── sources.py
│   └── util/
│       ├── network_utils.py
│       └── parsing_utils.py
├── scripts/               — CLI для датасета и экспериментов (см. scripts/README.md)
├── data/
│   ├── seeds/             — seed-статьи (elibrary_data_fixed.csv)
│   ├── processed/         — обучающий датасет
│   └── reports/           — кэши бенчмарков
├── models/                — model.pkl
├── tests/fixtures/        — HTML-фикстуры для тестов
└── docs/DESIGN_DOCUMENT.md
```

## Тесты

```bash
pytest
```

## Бенчмарки

```bash
python scripts/benchmark_rankers.py -n 20          # Crossref vs CyberLeninka vs Matcher
python scripts/compare_ranking.py                  # native vs ML по источникам
python scripts/compare_sources.py --english 15 --russian 15
```

## Ограничения

- Качество зависит от полноты запроса и наличия статьи в базах.
- OpenAlex участвует при inference; eLibrary доступен для датасета и бенчмарков (`sources=("...", "elibrary")`), но не в дефолтном `/match`.
