FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY src/ ./src/
# Unterbefehl wird beim Start angehängt, z. B. "docker compose run --rm app train"
ENTRYPOINT ["python", "-m", "src.main"]
CMD ["--help"]
