FROM python:3.10-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1


# Install build tools for scikit-surprise
RUN apt-get update && apt-get install -y build-essential


# Install poetry
RUN pip install poetry

# Copy all code and config
COPY . /app

# Install dependencies from pyproject.toml
RUN poetry install --no-interaction --no-ansi --only main

EXPOSE 8000

CMD ["poetry", "run", "uvicorn", "src.serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
