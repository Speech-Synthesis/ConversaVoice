# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file and install Python dependencies
COPY frontend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code into the container
COPY frontend/ ./frontend

# Define the command to run the application
CMD ["streamlit", "run", "frontend/app.py", "--server.port", "80", "--server.address", "0.0.0.0"]
