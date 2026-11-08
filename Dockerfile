FROM python:3.12.16-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TIKTOKEN_CACHE_DIR=/opt/tiktoken

WORKDIR /app

COPY requirements.lock ./
RUN pip install --no-cache-dir --disable-pip-version-check --root-user-action=ignore -r requirements.lock

# Nothing can be downloaded at run time, and the root filesystem is read-only:
# fetch the tokenizer data now and compile bytecode ahead of the first import.
RUN python -c "import tiktoken; [tiktoken.get_encoding(n) for n in ('cl100k_base', 'o200k_base')]"
# Some third-party test fixtures do not compile; that is harmless.
RUN python -m compileall -q /usr/local/lib/python3.12/site-packages || true

COPY run_agent.py ./
COPY harness ./harness
RUN python -m compileall -q /app
