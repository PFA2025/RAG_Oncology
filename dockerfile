FROM python:3.11-slim

# Environment variables for cleaner logging and Python path setup
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src

WORKDIR /app

# Copy only requirements.txt first (for caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy only the src folder into the container
COPY src ./src
COPY .env .
EXPOSE 8000

CMD bash -c "uvicorn server.app:app --host 0.0.0.0 --port 8000"