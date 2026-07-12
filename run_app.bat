@echo off
setlocal

:: Set Python to output unbuffered logs to prevent console lockups
set PYTHONUNBUFFERED=1

cd /d "%~dp0"

echo Cleaning up any existing servers running on ports 8000 (FastAPI) and 8501 (Streamlit)...
for /f "tokens=5" %%a in ('netstat -aon 2^>^&1 ^| findstr :8000 ^| findstr LISTENING') do (
  echo Terminating existing process %%a on port 8000...
  taskkill /f /pid %%a >nul 2>&1
)
for /f "tokens=5" %%a in ('netstat -aon 2^>^&1 ^| findstr :8501 ^| findstr LISTENING') do (
  echo Terminating existing process %%a on port 8501...
  taskkill /f /pid %%a >nul 2>&1
)

echo Clearing previous database and processed documents...
if exist "chroma_db" (
  rd /s /q "chroma_db"
)

echo Setting up Python virtual environment...
if not exist ".venv" (
  echo Creating virtual environment in .venv...
  python -m venv .venv
  if errorlevel 1 (
    echo [ERROR] Failed to create virtual environment. Make sure Python is installed.
    exit /b 1
  )
)

echo Activating virtual environment...
call .venv\Scripts\activate.bat
if errorlevel 1 (
  echo [ERROR] Failed to activate virtual environment.
  exit /b 1
)

echo Checking if python dependencies are satisfied...
python -c "import fastapi, uvicorn, langchain, langchain_community, langchain_chroma, langchain_huggingface, langchain_groq, langchain_google_genai, langchain_experimental, sentence_transformers, streamlit, psycopg2, dotenv, rank_bm25" >nul 2>&1
if errorlevel 1 (
  echo [INFO] Missing or outdated dependencies detected. Installing/upgrading packages...
  python -m pip install --upgrade pip >nul 2>&1
  python -m pip install -r requirements.txt
  if errorlevel 1 (
    echo [ERROR] Failed to install dependencies.
    exit /b 1
  )
) else (
  echo [SUCCESS] All dependencies are already satisfied! Skipping package installation.
)

:: Ensure the data directory exists
if not exist "data" (
  echo Creating "data" folder for PDF files...
  mkdir "data"
)

if exist "chroma_db" (
  echo Found existing Chroma database. Skipping ingestion.
) else (
  echo No Chroma database found. Running ingestion...
  python ingest.py
  if errorlevel 1 (
    echo [ERROR] Ingestion failed.
    exit /b 1
  )
)

echo Starting FastAPI backend in a background window...
start "Advanced RAG Backend" /min "%~dp0.venv\Scripts\python.exe" -u -m uvicorn backend.main:app --host 127.0.0.1 --port 8000

echo Starting Streamlit frontend app...
"%~dp0.venv\Scripts\python.exe" -m streamlit run app.py --server.port=8501 --server.address=127.0.0.1

endlocal
