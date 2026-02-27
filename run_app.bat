@echo off
setlocal

cd /d "%~dp0"

echo Loading from .env...

echo Installing Python dependencies...
python -m pip install --upgrade pip >nul 2>&1
python -m pip install -r requirements.txt
if errorlevel 1 (
  echo [ERROR] Failed to install dependencies.
  exit /b 1
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

echo Starting Streamlit app...
python -m streamlit run app.py

endlocal
