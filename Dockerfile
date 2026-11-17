FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    NUMBER_SEARCH_MODEL_DIR=/app/models/v5

RUN apt-get update \
    && apt-get install -y --no-install-recommends libglib2.0-0 libgomp1 fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 appuser

WORKDIR /app
COPY pyproject.toml README.md requirements.lock ./
RUN python -m pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu
COPY src/ src/
RUN python -m pip install -c requirements.lock .
COPY models/v5/ models/v5/
USER appuser
ENTRYPOINT ["number-search"]
CMD ["--help"]
