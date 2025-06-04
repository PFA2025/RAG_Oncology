FROM python:3.11-slim

# Environment variables for cleaner logging and Python path setup
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/cancer_agent/src

# Set working directory inside the container
WORKDIR /app

# Copy only requirements.txt first (for caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the cancer_agent directory into the container
COPY cancer_agent ./cancer_agent
COPY .env .

# Expose FastAPI port
EXPOSE 8000

# Start the app 
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000"]
