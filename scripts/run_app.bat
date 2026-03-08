@echo off
cd /d "%~dp0.."

echo Installing Backend dependencies...
pip install -r backend/requirements.txt
if %errorlevel% neq 0 (
    echo Failed to install backend dependencies.
    pause
    exit /b %errorlevel%
)

echo Installing Frontend dependencies...
pip install -r frontend/requirements.txt
if %errorlevel% neq 0 (
    echo Failed to install frontend dependencies.
    pause
    exit /b %errorlevel%
)

echo Starting Backend API...
start "ConversaVoice Backend" cmd /k "cd backend && uvicorn main:app --reload --port 8000"

echo Waiting for backend to initialize...
timeout /t 5

echo Starting Frontend...
streamlit run frontend/simulation_app.py
