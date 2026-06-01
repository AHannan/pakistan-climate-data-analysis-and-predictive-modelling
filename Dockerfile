# Using a lightweight Python base image
FROM python:3.9-slim

# Setting 'Working Directory' inside the container
WORKDIR /app

# Installing System Dependencies
RUN apt-get update && apt-get install -y \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency list first (for better caching)
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the Application Code and Model Files
COPY fastapi_app.py .
COPY *.joblib .

# Expose the port FastAPI will be running
EXPOSE 8000

# Command To run the application
CMD ["uvicorn", "fastapi_app:app", "--host", "0.0.0.0", "--port", "8000"]