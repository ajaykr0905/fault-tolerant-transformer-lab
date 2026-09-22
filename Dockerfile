FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
COPY configs ./configs
RUN pip install --no-cache-dir . \
    && addgroup --system app \
    && adduser --system --ingroup app app \
    && chown -R app:app /app

USER app
ENTRYPOINT ["fttl-train-smoke"]
CMD ["--config", "configs/smoke.json", "--output", "/tmp/fttl-smoke"]
