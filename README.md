# Citation Matcher

Программный инструмент для исправления ошибок в библиографических описаниях (тема 8).

Для заданного библиографического описания инструмент параллельно обращается к двум открытым наукометрическим базам — [Crossref](https://www.crossref.org/) и [OpenAlex](https://openalex.org/) — объединяет кандидатов, ранжирует их ML-моделью и возвращает наиболее вероятную статью с вероятностью совпадения и оформленной библиографической ссылкой.

## Установка

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -e .
```

## Обучение модели

Перед первым запуском обучите модель на подготовленном датасете:

```bash
python -m citation_matcher.train
```

Модель сохраняется в `models/model.pkl`.

## Использование

### Веб-интерфейс

```bash
uvicorn citation_matcher.app:app --reload
```

Откройте [http://localhost:8000](http://localhost:8000) в браузере.

### Командная строка

```bash
# Текстовый вывод
python -m citation_matcher "Attention is all you need Vaswani 2017"

# JSON-ответ
python -m citation_matcher --json "Deep learning LeCun Nature 2015"
```

Пример ответа:

```
Best match (probability: 39.22%):
LeCun Y., Bengio Y., Hinton G. Deep learning. Nature. 2015. Vol. 521 (7553). P. 436-444. DOI: 10.1038/nature14539

DOI: 10.1038/nature14539

Alternatives:
  0.65% — Deep learning & convolutional networks
  0.14% — Guest Editorial: Deep Learning
```

### Python API

```python
from citation_matcher import match_citation

result = match_citation("Deep learning LeCun Nature 2015")
print(result.best_match["confidence"])           # 0.98 — относительная уверенность
print(result.best_match["probability"])          # сырой ML-score (для отладки)
print(result.best_match["formatted_citation"])   # LeCun Y. ...
```

### JSON API

```bash
curl -X POST http://localhost:8000/api/match \
  -H "Content-Type: application/json" \
  -d '{"query": "Deep learning LeCun Nature 2015"}'
```

## Как это работает

1. **Поиск** — параллельные запросы в Crossref и OpenAlex; результаты объединяются, дубликаты по DOI удаляются.
2. **Признаки** — для каждого кандидата: сходство заголовка (`token_sort_ratio`, `token_set_ratio`), сходство первого автора, score базы, ранг, разница лет.
3. **Ранжирование** — `LogisticRegression.predict_proba` → вероятность совпадения.
4. **Результат** — лучший кандидат с вероятностью и оформленной ссылкой.

## Структура проекта

```
src/citation_matcher/
  search.py       — Crossref + OpenAlex API, объединение кандидатов
  features.py     — признаки для ML
  ranker.py       — загрузка модели и скоринг
  format.py       — форматирование библиографической ссылки
  matcher.py      — главная функция match_citation()
  cli.py          — командная строка
  app.py          — веб-интерфейс (FastAPI)
  train.py        — обучение модели
notebooks/          — исследование и сбор датасета
data/               — датасет и метрики baseline
models/             — обученная модель (model.pkl)
tests/              — unit-тесты
```

## Качество

Baseline Crossref (без ML) на синтетическом датасете из 480 запросов — `data/summary.csv`:

| Тип ошибки              | Top-1 | Top-3 |
|-------------------------|-------|-------|
| Без ошибок              | 96.9% | 100%  |
| Без автора              | 93.8% | 97.9% |
| Без года                | 95.8% | 100%  |
| Опечатка в заголовке    | 94.8% | 96.9% |
| Пропущенное слово       | 95.8% | 97.9% |

ML-модель (Accuracy на тесте: **99%**) переупорядочивает объединённых кандидатов из обеих баз.

## Тесты

```bash
pytest
```

## Ограничения

- Оптимальная работа — для статей с DOI в Crossref или OpenAlex.
- Качество зависит от полноты запроса (заголовок, автор, год).
