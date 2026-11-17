# Number Search

[Полный отчёт о работе и тестировании](REPORT.md).

Распознавание российского автомобильного номера по JPG/PNG. На входе — фотография,
на выходе — текст, полигон номера и служебные оценки. JSON-разметка для предсказания
не требуется. В поставке один рабочий конвейер v5: CNN-детектор, два распознавателя
с вниманием и консервативное уточнение глифовыми CNN.

**Проверенное качество: 84% — 42/50 на заранее отложенной внешней выборке.**
Это ограничение модели: цель 90% пока не достигнута. Упаковка для эксплуатации
не меняет её качество. Подробности: [карточка модели](docs/model-card.md).

## Установка

Python 3.12+, проверено локально на Python 3.14 / Windows, CPU.

```powershell
python -m venv .venv
.venv/Scripts/python.exe -m pip install -c requirements.lock -e .
.venv/Scripts/python.exe -m number_search verify-models
```

На Linux/macOS интерпретатор окружения — `.venv/bin/python`.
После установки также доступна команда `number-search`.
Выбранные веса лежат в [models/v5](models/v5); контрольные суммы проверяются при загрузке.

## Распознавание

```powershell
.venv/Scripts/python.exe -m number_search predict --image data/test/colored/a350aa150.jpg
.venv/Scripts/python.exe -m number_search predict --image photo.jpg --output outputs/prediction.json --visualize-dir outputs/prediction
```

JSON выводится в stdout, сообщения об ошибках — в stderr. Код завершения:
`0` — команда выполнена, `2` — ошибка ввода, конфигурации или файлов.
Непрочитанный номер возвращается пустой строкой и не считается ошибкой процесса.

Путь к модели: `--model-dir`, затем `NUMBER_SEARCH_MODEL_DIR`, затем локальный
`models/v5`. Во внешней установке wheel модель задаётся явно или размещается
в `models/v5` рабочего каталога. Количество CPU-потоков CLI: `--threads 4`.

## Python API

```python
from number_search import Recognizer

recognizer = Recognizer(model_dir="models/v5", threads=4)
result = recognizer.predict("photo.jpg")
print(result["text"])
```

Экземпляр загружает веса один раз и используется для многих фотографий.
`predict_array()` принимает RGB-массив NumPy `uint8` формы `(H, W, 3)`.
При передаче `threads` изменяется глобальный размер пула PyTorch в процессе.

## Оценка и визуализация

```powershell
.venv/Scripts/python.exe -m number_search evaluate --manifest data/external/holdout/manifest.json --report outputs/holdout.json --visualize-dir outputs/holdout
.venv/Scripts/python.exe -m number_search evaluate --split data/test/colored --report outputs/local-test.json --visualize-dir outputs/local-test
```

Manifest отделяет обезличенное имя изображения от эталонной строки. В режиме
`--split` эталон берётся из имени файла. Эталоны используются только после
предсказания, не передаются нейросети. Визуализация сохраняет контуры,
прочитанный текст, эталоны, фрагменты номеров, `summary.jpg` и `index.html`.

[Готовая галерея 50 изображений](reports/holdout/visualizations/index.html) ·
[Результаты](reports/holdout/results.json) · [История разработки](docs/research.md)

После рефакторинга все 50 строк совпадают с исходным протоколом v5.
Повторный запуск этого известного набора — регрессионная проверка,
а не новая независимая оценка.

## Структура

```text
src/number_search/       библиотека и CLI
  networks.py           выбранные архитектуры
  predictor.py          публичный Recognizer
  pipeline.py           области, согласование и восстановление
  reader.py             внимание и уточнение глифами
  training/             отдельный офлайн-контур обучения/скачивания
models/v5/              единственный комплект весов, конфигурация, хеши
tests/                  unit и интеграционные проверки
data/train,valid,test/  исходные пользовательские данные, сохранены локально
data/external/holdout/  50 контрольных изображений и manifest
data/manifests/         состав расширенного обучающего набора
data/sources/           источники и лицензии внешних данных
reports/                результаты и происхождение выбранной модели
docs/                   архитектура, эксплуатация и обучение
tools/                  упаковка нового выпуска весов
```

Из репозитория убраны CTC/GRU и эксперимент направления обхода, альтернативные
точки входа, разовые скрипты, промежуточные веса и скачанные архивы.
Глифовые CNN сохранены: они входят в действующую v5.
Исходные фотографии пользователя не удалялись.

## Разработка и сборка

```powershell
.venv/Scripts/python.exe -m pip install -c requirements.lock -e ".[dev]"
.venv/Scripts/python.exe -m ruff check src tests tools
.venv/Scripts/python.exe -m ruff format --check src tests tools
.venv/Scripts/python.exe -m pytest --run-integration
.venv/Scripts/python.exe -m build
```

Без `--run-integration` выполняются проверки без загрузки рабочего набора весов.
GitHub Actions проверяет форматирование, тесты и сборку на Python 3.12/3.14.
Сам workflow добавлен в репозиторий; удалённый CI из этой сессии не запускался.

Эксплуатация и Docker: [docs/operations.md](docs/operations.md).
Обучение и восстановление удалённых кешей: [docs/training.md](docs/training.md).
