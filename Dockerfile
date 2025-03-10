# Use official Python image with Alpine for smaller size
FROM python:3.10-alpine

# Set the working directory
WORKDIR /app

# Install system dependencies required for OpenCV and other libraries
RUN apk add --no-cache \
    build-base \
    cmake \
    ninja \
    python3-dev \
    gcc \
    g++ \
    make \
    musl-dev \
    libjpeg-turbo-dev \
    zlib-dev \
    freetype-dev \
    lcms2-dev \
    libwebp-dev \
    tiff-dev \
    tcl-dev \
    tk-dev \
    openjpeg-dev \
    git \
    opencv \
    libstdc++

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the application files
COPY . .

# Expose the FastAPI port
EXPOSE 5000

# Command to run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "5000", "--reload"]
