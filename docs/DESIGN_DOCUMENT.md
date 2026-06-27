# Citation Matcher — дизайн-документ

Сопоставление «шумных» библиографических описаний со статьями в открытых базах (Crossref, OpenAlex, CyberLeninka, eLibrary.ru).

## Содержание

1. [Команда](#1-команда)
2. [Объект исследования](#2-объект-исследования)
3. [Цель работы](#3-цель-работы)
4. [Методы исследования](#4-методы-исследования)
5. [Конкуренты и аналоги](#5-конкуренты-и-аналоги)
6. [Функциональности](#6-функциональности)
7. [Библиотеки](#7-библиотеки)
8. [MVP](#8-mvp)
9. [Результаты бенчмарка](#9-результаты-бенчмарка)
10. [Разделение обязанностей](#10-разделение-обязанностей)
11. [Архитектура и структура репозитория](#11-архитектура-и-структура-репозитория)
12. [Источники](#12-источники)

---

## 1. Команда

*(указать ФИО участников)*

---

## 2. Объект исследования

Задача автоматического сопоставления библиографических описаний с опечатками, без автора, без года и с пропущенными словами — с реальными научными публикациями в открытых наукометрических базах.

**Датасет:** пары «запрос — список кандидатов» с разметкой правильной статьи; до 10 кандидатов на запрос, 5 типов искусственных ошибок.

---

## 3. Цель работы

1. Построить сервис `/match`: по описанию — наиболее вероятная статья, калиброванная вероятность, оформленная ссылка.
2. Сравнить нативное ранжирование баз (Crossref, CyberLeninka) с ML-ранжированием поверх объединённого пула.
3. Оценить прирост Top-1 / Top-3 при типичных ошибках в описании.

---

## 4. Методы исследования

### 4.1. Мульти-источниковый поиск

Параллельные запросы в четыре API; дедупликация кандидатов; сохранение `source_rank` — позиции в ответе конкретной базы.

### 4.2. Признаки (feature engineering)

| Признак | Описание |
|---------|----------|
| `title_similarity` | Нечёткое сходство заголовков |
| `title_token_set_similarity` | Token-set ratio |
| `first_author_similarity` | Сходство первого автора |
| `year_difference` | Разница годов |
| `journal_similarity` | Сходство журнала |
| `word_count` | Число слов в запросе |
| `script_overlap` | Jaccard по алфавитам (latin / cyrillic / arabic) |

### 4.3. Обучение ранжировщика

LogisticRegression (`class_weight=balanced`) + калибровка (`CalibratedClassifierCV`, sigmoid). Негативы — остальные кандидаты в том же списке (≈10:1).

### 4.4. Inference

- Нормализация `predict_proba` внутри одного запроса (сумма ≈ 100%).
- Fallback на `title_token_set_similarity`, если все скоры ≈ 0.
- Boost до 99% при exact match заголовка.

### 4.5. Бенчмарки

| Скрипт | Назначение |
|--------|------------|
| `scripts/benchmark_rankers.py` | N статей; Crossref vs CyberLeninka vs Matcher; Top-1 / Top-3 по ошибкам |
| `scripts/compare_ranking.py` | Native vs ML внутри источника |
| `scripts/compare_sources.py` | Общий пул EN+RU; все базы + Matcher |
| `scripts/build_dataset.py` | Сбор обучающего датасета |

Код бенчмарков: `src/citation_matcher/benchmark/` (`evaluate`, `rankers`, `sources`).

**Метрики:** Top-1, Top-3, Recall@10, MRR.

---

## 5. Конкуренты и аналоги

| | **Citation Matcher** | **Нативный Crossref** | **Google Scholar** |
|---|---------------------|----------------------|-------------------|
| Назначение | Исправление ошибок + ранжирование + вероятность | Поиск по метаданным API | Ручной поиск |
| Источники | Crossref, OpenAlex, CyberLeninka, eLibrary | Только Crossref | Агрегатор (закрытый API) |
| ML-ранжирование | Да, 7 признаков | Нет | Нет |
| Устойчивость к опечаткам | Да (synthetic typos) | Слабая | Зависит от запроса |
| Русскоязычные статьи | CyberLeninka + eLibrary | Ограниченно | Хорошо, без API |
| Вероятность совпадения | Да, нормированная | Нет | Нет |
| API / UI | FastAPI + CLI + Python | REST API | Только браузер |
| Воспроизводимые бенчмарки | Да | Нет | Нет |

---

## 6. Функциональности

### 6.1. Поиск и агрегация

- Crossref (`query.bibliographic`), OpenAlex, CyberLeninka (JSON), eLibrary (HTML + cookies).
- Дедупликация по DOI / URL / нормализованному заголовку.
- Отказоустойчивость: сбой eLibrary не блокирует остальные источники.

### 6.2. ML-ранжирование (`/match`)

- Признаки на каждого кандидата → калиброванная вероятность → нормализация.
- Библиографическая ссылка (авторы, журнал, год, DOI/URL).

### 6.3. Датасет и обучение

- Seed-статьи: Crossref, CyberLeninka; опционально eLibrary (`data/seeds/elibrary_data_fixed.csv`).
- 5 типов ошибок: без ошибок, опечатка, без автора, без года, пропущенное слово.
- `python -m citation_matcher.train` → `models/model.pkl`.

### 6.4. Интерфейсы

| Интерфейс | Команда |
|-----------|---------|
| CLI | `python -m citation_matcher "запрос"` |
| Web UI | `uvicorn citation_matcher.app:app` |
| Python API | `from citation_matcher import match_citation` |

---

## 7. Библиотеки

| Для чего | Библиотека | Альтернативы | Почему |
|----------|------------|--------------|--------|
| HTTP, eLibrary | `curl-cffi`, `requests` | `httpx` | Обход anti-bot eLibrary |
| HTML eLibrary | `beautifulsoup4` | regex | Разбор таблиц результатов |
| Строки | `rapidfuzz` | `fuzzywuzzy` | Скорость, лицензия |
| ML | `scikit-learn` | XGBoost | Калибровка из коробки |
| Данные | `pandas` | Polars | CSV-пайплайн |
| Модель | `joblib` | pickle | Стандарт sklearn |
| Веб | `fastapi`, `uvicorn`, `jinja2` | Flask | Лёгкий API |
| Тесты | `pytest` | unittest | Краткие unit-тесты |

---

## 8. MVP

**Цель:** CLI по описанию → Crossref + CyberLeninka → лучший кандидат.

| Компонент | Статус |
|-----------|--------|
| Crossref + CyberLeninka | ✓ |
| Базовые признаки | ✓ |
| LogisticRegression | ✓ (заменено на калиброванную) |
| CLI | ✓ |

**Вне MVP (реализовано позже):** OpenAlex, eLibrary, FastAPI, нормализация вероятностей, бенчмарки, `script_overlap`, двуязычный пул.

---

## 9. Результаты бенчмарка

*N = 10 случайных статей, 50 запросов (кэш: `data/reports/ranker_benchmark.csv`)*

### Top-1

| Ошибка | Crossref | CyberLeninka | Matcher |
|--------|----------|--------------|---------|
| Без ошибок | 50.0% | 10.0% | **60.0%** |
| Опечатка в заголовке | 50.0% | 0.0% | **50.0%** |
| Без автора | 50.0% | 20.0% | **80.0%** |
| Без года | 50.0% | 20.0% | **70.0%** |
| Пропущенное слово | 50.0% | 10.0% | **60.0%** |
| **Все** | **50.0%** | **12.0%** | **64.0%** |

### Top-3

| Ошибка | Crossref | CyberLeninka | Matcher |
|--------|----------|--------------|---------|
| Все | 50.0% | 14.0% | **64.0%** |

Воспроизведение: `python scripts/benchmark_rankers.py -n 10 --refresh`

---

## 10. Разделение обязанностей

| Участник | Зона |
|----------|------|
| … | API (Crossref, OpenAlex, CyberLeninka, eLibrary) |
| … | Feature engineering, обучение |
| … | FastAPI, CLI |
| … | Бенчмарки, тесты, документация |

---

## 11. Архитектура и структура репозитория

### Поток данных

```
Запрос → multi_search() → build_features() → Ranker → normalize → MatchResult
              │
              ├── Crossref, OpenAlex, CyberLeninka, eLibrary
```

### Дерево проекта

```
citation-matcher/
├── src/citation_matcher/          # основной пакет
│   ├── config.py                  # пути, API, константы
│   ├── search.py                  # мульти-поиск
│   ├── elibrary.py                # клиент eLibrary.ru
│   ├── matcher.py                 # признаки, ранжирование, match_citation
│   ├── dataset.py                 # генерация обучающих примеров
│   ├── train.py                   # обучение модели
│   ├── app.py                     # FastAPI UI
│   ├── benchmark/                 # офлайн-оценка (не runtime)
│   │   ├── evaluate.py            # native vs ML по датасету
│   │   ├── rankers.py             # Crossref vs CyberLeninka vs Matcher
│   │   └── sources.py             # двуязычный пул, все источники
│   └── util/
│       ├── network_utils.py
│       └── parsing_utils.py
├── scripts/                       # CLI-обёртки для экспериментов
│   ├── build_dataset.py
│   ├── benchmark_rankers.py
│   ├── compare_ranking.py
│   └── compare_sources.py
├── data/
│   ├── seeds/                     # seed-статьи (eLibrary CSV)
│   ├── processed/                 # ranking_dataset*.csv
│   └── reports/                   # кэши бенчмарков
├── models/model.pkl
├── tests/
│   └── fixtures/                  # HTML-фикстуры для парсинга
└── docs/DESIGN_DOCUMENT.md
```

---

## 12. Источники

1. Crossref REST API — https://api.crossref.org/
2. OpenAlex API — https://docs.openalex.org/
3. Loper, E., Bird, S. NLTK // ACL Workshop. — 2002.
4. Platt, J. Probabilistic Outputs for SVMs // Advances in Large Margin Classifiers. — 1999.
5. Cohen, W. W. et al. String Distance Metrics for Name-Matching // IIWeb. — 2003.
6. scikit-learn: Probability calibration — https://scikit-learn.org/stable/modules/calibration.html
