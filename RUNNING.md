
You have two options to run the application:

### Option 1: Development Mode (Two Terminals)

**Terminal 1 (Backend):**
```powershell
cd backend
uvicorn main:app --reload --port 8000
```

**Terminal 2 (Frontend):**
```powershell
streamlit run app.py
```

### Option 2: Production Mode (Using Script)

**Windows:**
Double-click `run_app.bat`

**Linux/Mac:**
Run `./run_app.sh`
