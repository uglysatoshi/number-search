# Офлайн-обучение

Рабочий инференс не зависит от этого контура. Подготовка данных и обучение
запускаются только явными CLI-подкомандами.

## Восстановление расширенных данных

В `data/manifests/training.json` сохранены immutable source revisions,
имена элементов архивов, SHA-256, текст/координаты и пути кеша.
Исходные ZIP и извлечённые обучающие картинки удалены при очистке.
Их можно восстановить:

```bash
number-search prepare-data --manifest data/manifests/training.json --workers 8
```

Скачивание использует HTTP Range, читает только выбранные элементы source
train/val ZIP, проверяет CRC и SHA-256. Готовые правильные файлы пропускаются.
Запись ограничена `data/cache` относительно каталога manifest; исходные
`data/train`, `valid`, `test` не перезаписываются.
В зависимости от доступности источника и сети подготовка может занять время.

## Обучение выбранных архитектур

```bash
number-search train detector --manifest data/manifests/training.json --output runs/detector --epochs 35 --lr 0.0003 --include-local data/train
number-search train reader --manifest data/manifests/training.json --output runs/reader-a --epochs 25 --lr 0.0008 --seed 941
number-search train reader --manifest data/manifests/training.json --output runs/reader-b --epochs 15 --lr 0.0005 --strong-geometry --seed 943 --resume runs/reader-a/sequence.pt
number-search train glyphs --manifest data/manifests/training.json --output runs/glyphs --epochs 15
```

Без `--resume` веса инициализируются с нуля. С ним загружается совместимый
checkpoint текущей архитектуры; состояние оптимизатора создаётся заново.
`--include-local` нужен только детектору и добавляет исходные LabelMe-полигоны.
Глифы автоматически выделяются из OCR-строк; совпадение числа компонент
с длиной строки — эвристика, поэтому качество таких меток требует проверки.

Все стадии сохраняют best checkpoint по validation и `history.json`.
У детектора критерий — IoU, у читателя — exact accuracy, затем CER;
у глифов — accuracy, затем loss. Seed, размер batch и число потоков задаются
параметрами CLI. При одинаковых данных побитовая повторяемость между разными
библиотеками и оборудованием не обещается.

Эти команды задают поддерживаемый процесс будущего обучения. Они не обещают
воспроизведения уже выпущенных v5-весов: при их создании использовались
собственные прежние и промежуточные веса, которые не нужны инференсу и удалены.
Выбранные v5 checkpoint сохранены неизменными, их происхождение и истории — в reports.

## Защита разбиения

Обучающий manifest хранит исключённые тексты контрольных номеров. Загрузчик
проверяет пересечения OCR train/validation/blocked и хеши detector train/validation.
Новый независимый тест нужно отложить до выбора checkpoint и алгоритма.
Изученные 50 изображений теперь полезны для регрессии, но не для следующего
доказательства независимой точности.

## Сборка релиза

Избранные `detector.pt`, `sequence.pt`, `sequence_aux.pt`, `characters.pt`,
`characters_aux.pt` помещаются в отдельный staging-каталог. Затем вызывается
`tools/package_models.py`, как описано в [operations.md](operations.md).
Конфигурация и хеши — часть выпуска. Не редактируйте уже проверенные веса
и `pipeline.json` без создания нового выпуска.
