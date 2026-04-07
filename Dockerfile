FROM python:3.10-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install only serving-time dependencies to keep image lean.
RUN pip install --no-cache-dir --upgrade pip && \
	pip install --no-cache-dir fastapi==0.115.6 uvicorn[standard]==0.34.0 pandas==2.2.3 numpy==1.26.4 pyarrow==20.0.0 pyyaml==6.0.2 scikit-learn==1.5.2

COPY . /app

EXPOSE 8000

CMD ["uvicorn", "src.serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
