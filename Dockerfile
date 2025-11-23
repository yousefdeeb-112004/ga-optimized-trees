# Multi-stage build for GA-Optimized Decision Trees
FROM python:3.10-slim as builder

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --user -r requirements.txt

# Final stage
FROM python:3.10-slim

WORKDIR /app

# Install runtime dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy Python packages from builder
COPY --from=builder /root/.local /root/.local

# Copy application code
COPY . .

# Install package in editable mode
RUN pip install --no-cache-dir -e .

# Create necessary directories
RUN mkdir -p /app/results /app/models /app/data /app/logs

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app/src:$PYTHONPATH
ENV PATH=/root/.local/bin:$PATH

# Default command (can be overridden)
CMD ["python", "scripts/train.py", "--help"]

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import ga_trees; print('OK')" || exit 1

# Labels
LABEL maintainer="your.email@example.com"
LABEL version="1.0.0"
LABEL description="GA-Optimized Decision Trees Framework"
