# Use an official Python runtime as a parent image
FROM python:3.10

# Set the working directory
WORKDIR /app

# Install system dependencies required for OpenCV
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install pip dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt uvicorn

# Copy the application code
COPY . .

# Ensure Uvicorn is in PATH
RUN echo "Checking uvicorn installation..." && \
    python -c "import uvicorn; print('Uvicorn installed successfully')"

# Expose the application's port
EXPOSE 5000

# Use explicit Python execution to avoid path issues
CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "5000"]
