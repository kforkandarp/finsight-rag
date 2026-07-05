# Use official Python 3.11 slim image — smaller than full Python image
FROM python:3.11-slim

# Set working directory inside container
WORKDIR /app

RUN pip install torch==2.1.0 --index-url https://download.pytorch.org/whl/cpu

# Copy requirements first — Docker caches this layer
# If requirements don't change, this layer isn't rebuilt on next build
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy rest of project
COPY . .

# Expose port 8000
EXPOSE 8000

# Command to run when container starts
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]