# API server image for Fly. The front end is built in a node stage; the GPU worker is deployed to Modal separately.
FROM node:22-slim AS web
WORKDIR /web
ENV PLAYWRIGHT_SKIP_BROWSER_DOWNLOAD=1
COPY web/package.json web/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY web/ ./
RUN npm run build

FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim
WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project
COPY src ./src
COPY worker/weights.json ./worker/weights.json
COPY fixtures ./fixtures
COPY --from=web /web/dist ./web/dist
RUN uv sync --frozen --no-dev
EXPOSE 8080
CMD ["/app/.venv/bin/python", "-m", "radar_desk"]
