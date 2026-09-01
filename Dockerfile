FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and config
COPY . .

# Ensure data and logs directories exist
RUN mkdir -p data logs results

# Default command: run the live paper trading daemon
CMD ["python", "scripts/run_live_paper.py"]
