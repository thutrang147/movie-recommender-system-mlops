FROM python:3.10-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1


# Install build tools for scikit-surprise
RUN apt-get update && apt-get install -y build-essential

# Install poetry and dependencies from pyproject.toml
RUN pip install poetry
COPY pyproject.toml poetry.lock* ./
RUN poetry install --no-interaction --no-ansi --only main

COPY . /app

EXPOSE 8000

CMD ["uvicorn", "src.serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
