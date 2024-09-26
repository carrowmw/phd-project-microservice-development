FROM python:3.10-slim-bookworm

ARG DATABASE_ENV

LABEL maintainer="{{ cookiecutter.project_domain }}"
LABEL vendor="{{ cookiecutter.project_domain }}"

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1 \
    PYTHONUNBUFFERED 1 \
    # pip:
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \


    # Install system dependencies
    RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Install poetry
RUN curl -sSL https://install.python-poetry.org | python3 -