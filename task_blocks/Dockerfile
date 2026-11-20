# No `# syntax=` directive on purpose. That line makes BuildKit fetch the
# `docker/dockerfile` frontend image from Docker Hub before it reads anything
# else, so a build fails outright when the registry is unreachable, even though
# every other layer is cached locally. Docker's built-in frontend already
# supports the `RUN --mount=type=cache` lines below, so the directive buys
# nothing here.

# Multi-stage build. The builder resolves and installs dependencies; the
# runtime image carries only the virtualenv and the application, so neither uv
# nor a compiler is present in the image that actually runs.
#
# Base images are pinned by digest, not by tag: `python:3.13-slim` moves, and a
# deployment that changes underneath you is not reproducible. Refresh the
# digests deliberately, with `docker inspect <image> --format '{{index
# .RepoDigests 0}}'`.
#
# `python:3.13-slim` is Debian-based and its sqlite3 is compiled with
# `SQLITE_ENABLE_LOAD_EXTENSION`, which the long-term store's vector index
# needs (`sqlite-vec` is a loadable extension). Alpine would need musl wheels
# for sqlite-vec; the macOS python.org and python-build-standalone builds
# cannot load extensions at all — see `memory.sqlite_supports_extensions`.

# --- builder ---------------------------------------------------------------
FROM python:3.13-slim@sha256:e48f59cdba69a3cbd9ed2cf0f348d90d45a8d350d98134cf1070bc106ee9ed90 AS builder

COPY --from=ghcr.io/astral-sh/uv:0.13.1@sha256:4a1f3666816dafadd20ceb71eac9a7d56d7ef30279f7d64048519ab901ada336 /uv /usr/local/bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /app

# Dependencies first, in their own layer: they change far less often than the
# source, so editing a module does not reinstall the world. --frozen fails the
# build if uv.lock disagrees with pyproject.toml, which is what makes the image
# reproducible.
# README.md is listed as the project's `readme` in pyproject.toml, so the
# build backend needs it present to build the project itself.
COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --no-dev

# Then the project itself.
COPY src/ ./src/
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

# --- runtime ---------------------------------------------------------------
FROM python:3.13-slim@sha256:e48f59cdba69a3cbd9ed2cf0f348d90d45a8d350d98134cf1070bc106ee9ed90 AS runtime

# curl is for the container healthcheck. Nothing else is installed: no shell
# utilities a compromised tool call could lean on, no build toolchain.
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl=8.* \
    && rm -rf /var/lib/apt/lists/*

# Non-root, with no login shell and no home directory to write to. UID is
# fixed so the /data volume's ownership is stable across rebuilds.
RUN groupadd --gid 10001 app \
    && useradd --uid 10001 --gid app --no-create-home --shell /usr/sbin/nologin app

WORKDIR /app

COPY --from=builder --chown=root:root /app/.venv /app/.venv
COPY --chown=root:root src/ ./src/
# Runtime-loaded prompts and skills. Copied read-only: the agent reads these,
# and nothing in the system should be able to rewrite its own instructions.
COPY --chown=root:root agent/ ./agent/

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app/src \
    DATA_DIR=/data

# The volume mount point, owned by the runtime user: the image's filesystem is
# read-only at runtime, so this is the only writable path besides tmpfs /tmp.
RUN mkdir -p /data && chown app:app /data

USER app
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD curl -fsS http://localhost:8000/health || exit 1

CMD ["uvicorn", "task_blocks.api:app", "--host", "0.0.0.0", "--port", "8000"]
