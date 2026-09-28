# syntax=docker/dockerfile:1
#
# Two independent images from one file: `--target api` for the FastAPI backend and
# `--target web` for the Next.js frontend. They scale separately and have different
# runtimes, so a single combined image would be the wrong unit of deployment.

# ---------- backend ----------

FROM python:3.12-slim AS api-base
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1
WORKDIR /app

FROM api-base AS api-builder
RUN pip install --no-cache-dir hatchling
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip wheel --no-deps --wheel-dir /wheels .

FROM api-base AS api
COPY --from=api-builder /wheels /wheels
RUN pip install --no-cache-dir /wheels/*.whl \
      "pydantic>=2.7" "fastapi>=0.110" "uvicorn>=0.27" \
    && rm -rf /wheels

COPY data ./data

# The document library is built into the image, so a container answers its first search
# without an indexing step.
ENV ANALYZER_LIBRARY_DIR=/app/library
RUN analyzer index data/corpus.jsonl --out "$ANALYZER_LIBRARY_DIR"

RUN useradd --create-home --uid 10001 analyzer && chown -R analyzer:analyzer /app
USER analyzer

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=2).status == 200 else 1)"

ENTRYPOINT ["analyzer"]
CMD ["serve", "--library", "/app/library", "--host", "0.0.0.0", "--port", "8000"]

# ---------- frontend ----------

FROM node:20-slim AS web-deps
WORKDIR /app
COPY web/package.json web/package-lock.json ./
# `npm ci` rather than `npm install`: the lockfile is committed, and a build that resolves
# a different dependency tree than the one that was tested is not the same build.
RUN npm ci --no-audit --no-fund

FROM node:20-slim AS web-builder
WORKDIR /app
COPY --from=web-deps /app/node_modules ./node_modules
COPY web/ ./
ARG NEXT_PUBLIC_API_URL=http://localhost:8000
ENV NEXT_PUBLIC_API_URL=$NEXT_PUBLIC_API_URL
RUN npm run build

FROM node:20-slim AS web
WORKDIR /app
ENV NODE_ENV=production
COPY --from=web-builder /app/package.json /app/package-lock.json ./
COPY --from=web-builder /app/node_modules ./node_modules
COPY --from=web-builder /app/.next ./.next
COPY --from=web-builder /app/next.config.mjs ./

RUN useradd --create-home --uid 10001 web && chown -R web:web /app
USER web

EXPOSE 3000
ENV PORT=3000 HOSTNAME=0.0.0.0
CMD ["npm", "run", "start"]
