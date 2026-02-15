# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Install system dependencies required for pyaudio
RUN apt-get update && apt-get install -y --no-install-recommends 
    build-essential 
    portaudio19-dev 
    && rm -rf /var/lib/apt/lists/*

# ---- START DEBUGGING ----
# Verify that the portaudio library was installed
RUN echo "--- Verifying portaudio installation ---" && dpkg -l | grep portaudio || echo "Portaudio not found"
# ---- END DEBUGGING ----

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file and install Python dependencies
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code into the container
COPY backend/ ./backend
COPY src/ ./src

# ---- START DEBUGGING ----
# List all copied files to verify paths
RUN echo "--- Verifying file structure ---" && ls -R
# ---- END DEBUGGING ----

# Define the command to run the application
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "80"]
