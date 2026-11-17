# Разработка

Установка: `python -m pip install -c requirements.lock -e ".[dev]"`.
Используется src-layout и единый `pyproject.toml`, консольная команда задаётся
через project.scripts. Описание стандарта:
[Python Packaging User Guide](https://packaging.python.org/en/latest/guides/writing-pyproject-toml/).

Команды:

```bash
python -m ruff check src tests tools
python -m ruff format --check src tests tools
python -m pytest
python -m pytest --run-integration
python -m build
```

Unit-тесты не требуют скачивания данных и не обращаются в сеть.
Интеграция использует версионированные веса и 50 локальных контрольных фото.
Тесты обучения делают по одному шагу всех поддерживаемых стадий на временных
синтетических изображениях, не изменяя production-веса.

При изменении preprocessing или сети обязательно проверяйте полную регрессию.
Отчёты исследования хранятся в reports. Текущие результаты и временные файлы
следует записывать в outputs/runs, которые игнорируются Git.

`requirements.lock` фиксирует проверенные runtime-версии. PyTorch в Linux CI
и Docker устанавливается из официального CPU-index, чтобы не загружать CUDA.
Новый runtime-lock следует менять вместе с повторной регрессией и сборкой.

GitHub Actions использует официальные
[checkout](https://github.com/actions/checkout) и
[setup-python](https://github.com/actions/setup-python).
