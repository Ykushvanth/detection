# PowerShell script to deploy to Hugging Face Spaces
# Make sure you have created the Space on Hugging Face first!

param(
    [Parameter(Mandatory=$true)]
    [string]$HfUsername,
    
    [Parameter(Mandatory=$true)]
    [string]$SpaceName
)

Write-Host "=========================================" -ForegroundColor Cyan
Write-Host "Deploying to Hugging Face Spaces" -ForegroundColor Yellow
Write-Host "=========================================" -ForegroundColor Cyan
Write-Host ""

# Check if git is initialized
if (-not (Test-Path .git)) {
    Write-Host "Initializing git repository..." -ForegroundColor Yellow
    git init
}

# Check current git status
Write-Host "Checking git status..." -ForegroundColor Yellow
git status

# Add all files
Write-Host ""
Write-Host "Adding files to git..." -ForegroundColor Yellow
git add .

# Check if there are changes to commit
$status = git status --porcelain
if ($status) {
    Write-Host "Committing changes..." -ForegroundColor Yellow
    git commit -m "Deploy to Hugging Face Spaces"
} else {
    Write-Host "No changes to commit." -ForegroundColor Green
}

# Remove existing origin if it exists
$existingRemote = git remote get-url origin 2>$null
if ($existingRemote) {
    Write-Host "Removing existing remote..." -ForegroundColor Yellow
    git remote remove origin
}

# Add Hugging Face remote
$hfUrl = "https://huggingface.co/spaces/$HfUsername/$SpaceName"
Write-Host ""
Write-Host "Adding Hugging Face remote: $hfUrl" -ForegroundColor Yellow
git remote add origin $hfUrl

# Push to Hugging Face
Write-Host ""
Write-Host "Pushing to Hugging Face..." -ForegroundColor Yellow
Write-Host "You will be prompted for:" -ForegroundColor Cyan
Write-Host "  - Username: Your Hugging Face username" -ForegroundColor Cyan
Write-Host "  - Password: Your Hugging Face Access Token (NOT your password!)" -ForegroundColor Cyan
Write-Host ""
Write-Host "Get your token from: https://huggingface.co/settings/tokens" -ForegroundColor Yellow
Write-Host ""

git push origin main

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "=========================================" -ForegroundColor Green
    Write-Host "Deployment initiated successfully!" -ForegroundColor Green
    Write-Host "=========================================" -ForegroundColor Green
    Write-Host ""
    Write-Host "Monitor your deployment at:" -ForegroundColor Cyan
    Write-Host "https://huggingface.co/spaces/$HfUsername/$SpaceName" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Check the 'Logs' tab to see build progress." -ForegroundColor Yellow
    Write-Host "First build may take 15-30 minutes." -ForegroundColor Yellow
} else {
    Write-Host ""
    Write-Host "=========================================" -ForegroundColor Red
    Write-Host "Deployment failed!" -ForegroundColor Red
    Write-Host "=========================================" -ForegroundColor Red
    Write-Host ""
    Write-Host "Common issues:" -ForegroundColor Yellow
    Write-Host "1. Invalid credentials - check your access token" -ForegroundColor Yellow
    Write-Host "2. Space doesn't exist - create it first at https://huggingface.co/spaces" -ForegroundColor Yellow
    Write-Host "3. Network issues - check your internet connection" -ForegroundColor Yellow
}

