@echo off
echo Installing dependencies...
pip install -r backend/requirements.txt
if %errorlevel% neq 0 (
    echo Failed to install dependencies. Please run "pip install -r backend/requirements.txt" manually.
    pause
    exit /b %errorlevel%
)

echo Starting Backend API...
start "ConversaVoice Backend" cmd /k "cd backend && uvicorn main:app --reload --port 8000"

echo Waiting for backend to initialize...
timeout /t 5

echo Starting Frontend...
streamlit run app.py
