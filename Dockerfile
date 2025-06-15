FROM python:3.9-slim

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set working directory
WORKDIR /app

# Copy source code and requirements
COPY ./app /app
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Set Hugging Face cache environment variables
ENV HF_HOME=/app/hf_cache
ENV TRANSFORMERS_CACHE=/app/hf_cache
ENV TORCH_HOME=/app/hf_cache

# Create the cache directory and ensure permissions
RUN mkdir -p /app/hf_cache && chmod -R 777 /app/hf_cache

COPY . .

CMD ["gunicorn", "--workers", "1", "--timeout", "120", "--bind", "0.0.0.0:7860", "app:app"]