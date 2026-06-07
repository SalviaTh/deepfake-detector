# Use Python 3.10
FROM python:3.10-slim

# Install system dependencies for OpenCV and other tools
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install
# We assume requirements.txt is in the root directory
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the code
COPY . .

# Expose port (Render/Hugging Face usually use 8000 or 7860)
EXPOSE 8000

# Start the server using uvicorn
# We use the main.py entry point
CMD ["python", "backend/main.py"]
