# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file and install Python dependencies
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code into the container
COPY backend/ ./backend
COPY src/ ./src

# Define the command to run the application
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "80"]
