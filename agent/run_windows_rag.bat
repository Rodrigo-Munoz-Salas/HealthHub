@echo off
echo 🏥 Windows RAG System
echo ====================
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python is not installed or not in PATH
    echo Please install Python 3.8+ and try again
    pause
    exit /b 1
)

echo ✅ Python found
echo.

REM Change to the src directory
cd src

REM Run the Windows RAG system test
echo 🔧 Running Windows RAG System...
python test_windows_simple.py

if errorlevel 1 (
    echo.
    echo ❌ Test failed. Please check the errors above.
    echo.
    echo 💡 Troubleshooting:
    echo    1. Make sure all dependencies are installed: pip install -r requirements_windows.txt
    echo    2. Check that model files exist in mobile_models/ directory
    echo    3. Verify Python version is 3.8 or higher
    echo.
    pause
    exit /b 1
) else (
    echo.
    echo ✅ Windows RAG System is working correctly!
    echo.
    echo 💡 To run the full system:
    echo    python windows_rag_system.py
    echo.
    pause
)
