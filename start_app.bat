@echo off
title Book Recommendation System Launcher
echo ================================
echo  BOOK RECOMMENDATION SYSTEM
echo ================================
echo.

:: STEP 1: Check Python
where python >nul 2>nul
IF %ERRORLEVEL% NEQ 0 (
    echo ❌ Python is not installed! Please install Python from https://www.python.org/downloads/
    pause
    exit /b
)

:: STEP 2: Check if required packages are installed
pip show streamlit >nul 2>nul
IF %ERRORLEVEL% NEQ 0 (
    echo 📦 Installing required Python packages...
    python -m pip install --upgrade pip
    pip install streamlit pandas numpy scikit-learn matplotlib seaborn
) ELSE (
    echo ✅ Required packages already installed.
)

:: STEP 3: Run the Streamlit app
echo 🚀 Launching the app...
streamlit run app.py

:: STEP 4: Keep window open
pause
