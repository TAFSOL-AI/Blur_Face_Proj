# Use an official Python runtime as a parent image
FROM python:3.10

# Set the working directory
WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY . .

# Ensure uvicorn is installed
RUN pip install uvicorn

# Expose the application's port
EXPOSE 5000

# Set the command to run the FastAPI app
CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "5000"]
