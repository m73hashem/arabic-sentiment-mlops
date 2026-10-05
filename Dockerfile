FROM python:3.13.16-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HUB_OFFLINE=1 \
    HF_HOME=/tmp/huggingface \
    SENTIMENT_MODEL_DIR=/models/full-gpu-arabert-inference \
    PYTHONPATH=/app/src

ARG APP_UID=10001
ARG APP_GID=10001

RUN groupadd --gid "${APP_GID}" app \
    && useradd --uid "${APP_UID}" --gid app --no-create-home \
        --shell /usr/sbin/nologin app

WORKDIR /app

COPY requirements-api.txt /tmp/requirements-api.txt
RUN python -m pip install \
        --disable-pip-version-check \
        --no-cache-dir \
        --index-url https://pypi.org/simple \
        --extra-index-url https://download.pytorch.org/whl/cpu \
        -r /tmp/requirements-api.txt \
    && rm /tmp/requirements-api.txt

COPY --chown=app:app src/ /app/src/

USER app

EXPOSE 8000

CMD ["uvicorn", "arabic_sentiment.api:app", "--host", "0.0.0.0", "--port", "8000"]
