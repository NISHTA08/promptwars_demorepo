FROM python:3.11-slim

WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Start the application using uvicorn and dynamically bind to Cloud Run's $PORT
CMD exec uvicorn main:app --host 0.0.0.0 --port ${PORT:-8080}
