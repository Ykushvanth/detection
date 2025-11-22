@echo off
REM Batch script to build and run Docker container locally
REM This script builds and runs the Docker container for local testing

echo =========================================
echo Building Docker Image...
echo =========================================

REM Build the Docker image
docker build -t face-detection-api .

if errorlevel 1 (
    echo Docker build failed!
    exit /b 1
)

echo.
echo =========================================
echo Build successful!
echo =========================================
echo.
echo Starting container...
echo API will be available at: http://localhost:7860
echo API Docs will be available at: http://localhost:7860/docs
echo Health check: http://localhost:7860/health
echo.
echo Press Ctrl+C to stop the container
echo.

REM Run the container
docker run -p 7860:7860 face-detection-api


