# Скрипты

CLI-обёртки над пакетом `citation_matcher`. Запускать из корня репозитория.

| Скрипт | Назначение |
|--------|------------|
| `build_dataset.py` | Сбор обучающего датасета (Crossref, CyberLeninka, опционально eLibrary) |
| `benchmark_rankers.py` | Crossref vs CyberLeninka vs Matcher; Top-1 / Top-3 по типам ошибок |
| `compare_ranking.py` | Native vs ML внутри одного источника |
| `compare_sources.py` | Общий двуязычный пул; сравнение всех баз и Matcher |

Логика бенчмарков — в `src/citation_matcher/benchmark/`.

```bash
python scripts/build_dataset.py --crossref 50 --cyberleninka 50
python scripts/benchmark_rankers.py -n 20
python scripts/compare_ranking.py --group-by error
python scripts/compare_sources.py --english 15 --russian 15
```

Кэши результатов: `data/reports/`.
