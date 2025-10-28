FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p /app/chroma_db /app/logs

# Make startup script executable
RUN chmod +x /app/scripts/build_vector_store.py

# Set environment variable to disable tokenizers warning
ENV TOKENIZERS_PARALLELISM=false

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=60s --retries=3 \
    CMD python /app/scripts/build_vector_store.py health || exit 1

# Startup command with vector store initialization
CMD ["sh", "-c", "python /app/scripts/build_vector_store.py && uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload"]
