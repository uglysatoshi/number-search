# Эксплуатация

## Конфигурация и интерфейс

Устанавливаемый пакет предоставляет `number-search` и `python -m number_search`.
Предсказание не требует доступа к сети, training-кешам или разметкам.
Путь к комплекту весов задаётся `--model-dir` либо `NUMBER_SEARCH_MODEL_DIR`.
В исходном checkout по умолчанию используется `models/v5`.

Загрузка проверяет SHA-256 пяти checkpoint и `pipeline.json`, формат manifest,
параметры конфигурации и точное совпадение ключей state dict.
Используется `torch.load(..., weights_only=True)` на CPU. Отсутствующий,
повреждённый или несовместимый комплект приводит к явной ошибке.

CLI выводит JSON в stdout, ошибки и сообщения подготовки данных — в stderr.
`--verbose` перед подкомандой включает traceback для диагностики.
Запись JSON выполняется через временный файл и atomic replace.
Выходной отчёт не может перезаписать исходную фотографию/manifest.

Рекомендуется загружать один `Recognizer` на рабочий процесс и повторно
использовать его. `--threads` относится к PyTorch; число процессов и потоков
нужно выбирать под доступный CPU. В Python API `threads=None` сохраняет
настройки пула вызывающего приложения.

## Docker

```bash
docker build -t number-search:0.1.0 .
docker run --rm -v "$PWD/data/test:/input:ro" number-search:0.1.0 predict --image /input/colored/a350aa150.jpg
docker run --rm -v "$PWD/data/external/holdout:/input:ro" -v "$PWD/outputs:/output" number-search:0.1.0 evaluate --manifest /input/manifest.json --report /output/report.json --visualize-dir /output/gallery
```

Контейнер содержит только пакет, runtime-зависимости и выбранные модели.
Работает без root под UID 10001. Выходной volume должен быть доступен этому
пользователю для записи. В контекст сборки не попадают фотографии, кеши,
истории обучения и локальное окружение. Используется CPU-сборка PyTorch.

Docker недоступен в среде текущей проверки, поэтому фактическая сборка
образа здесь не запускалась. Python-пакет, CLI и wheel проверены локально.

## Проверка перед выпуском

```bash
python -m number_search verify-models --model-dir models/v5
python -m pytest --run-integration
python -m ruff check src tests tools
python -m ruff format --check src tests tools
python -m build
```

Интеграционная проверка сравнивает не только итоговую долю, но и каждый
из 50 текстов с `reports/holdout/original_results.json`.
CI выполняет проверки на Linux/Python 3.12 и 3.14; workflow не публикует образ
и не развёртывает приложение. Удалённый CI здесь не запускался.

## Выпуск новых весов

Весам обучения назначается новый каталог выпуска; текущий `models/v5`
не используется как output тренировки. После выбора и проверки пяти файлов:

```bash
python tools/package_models.py --weights runs/selected --config models/v5/pipeline.json --version v6 --output models/v6
```

Утилита создаёт manifest, проверяет загрузку и атомарно переносит готовый
комплект. Существующий выпуск не перезаписывается. Переключение выполняется
явным путём или переменной окружения; для отката указывается предыдущий каталог.
Wheel содержит код. Модели поставляются отдельным каталогом, а Dockerfile
включает выбранный v5. Архивы моделей должны передаваться вместе с manifest.

## Ограничение качества

v5 имеет 84% полного совпадения на сохранённой отложенной выборке.
Внутренние scores не заменяют контроль результата в бизнес-процессе.
Изменения кода и упаковки не являются дополнительным обучением модели.
