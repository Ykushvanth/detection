# PowerShell script to build and run Docker container locally
# This script builds and runs the Docker container for local testing

Write-Host "=========================================" -ForegroundColor Cyan
Write-Host "Building Docker Image..." -ForegroundColor Yellow
Write-Host "=========================================" -ForegroundColor Cyan

# Build the Docker image
docker build -t face-detection-api .

if ($LASTEXITCODE -ne 0) {
    Write-Host "Docker build failed!" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "=========================================" -ForegroundColor Green
Write-Host "Build successful!" -ForegroundColor Green
Write-Host "=========================================" -ForegroundColor Green
Write-Host ""
Write-Host "Starting container..." -ForegroundColor Yellow
Write-Host "API will be available at: http://localhost:7860" -ForegroundColor Cyan
Write-Host "API Docs will be available at: http://localhost:7860/docs" -ForegroundColor Cyan
Write-Host "Health check: http://localhost:7860/health" -ForegroundColor Cyan
Write-Host ""
Write-Host "Press Ctrl+C to stop the container" -ForegroundColor Yellow
Write-Host ""

# Run the container
docker run -p 7860:7860 face-detection-api


