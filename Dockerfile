# Use the official Python Alpine/Slim image for a smaller footprint
FROM python:3.12-slim

# Prevent Python from writing pyc files and keep stdout non-buffered
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install dependencies efficiently
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . /app/

# Port matching the .env specification
EXPOSE 8000

# Execute the FastAPI server natively
CMD ["python", "main.py", "serve"]
