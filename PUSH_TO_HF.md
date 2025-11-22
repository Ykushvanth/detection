# Push to Hugging Face - Authentication Required

## Issue
Your account needs authentication. You need to use an **Access Token** instead of password.

## Solution: Use Access Token

### Step 1: Get Your Access Token

1. Go to: https://huggingface.co/settings/tokens
2. Click "New token"
3. Name it: `face-detection-deploy`
4. Select **"Write"** permissions
5. Click "Generate token"
6. **Copy the token immediately** (you won't see it again!)

### Step 2: Push Using Access Token

Run these commands in PowerShell:

```powershell
cd C:\Users\Kushvanth\OneDrive\Desktop\Detection\FaceDetection

# Make sure remote is correct
git remote -v

# If not pointing to Hugging Face, fix it:
git remote remove origin
git remote add origin https://huggingface.co/spaces/Velugondaiah/FaceDetection

# Add all files
git add .

# Commit (if there are changes)
git commit -m "Deploy Face Detection API to Hugging Face"

# Push using token
git push origin main
```

**When prompted:**
- **Username**: `Velugondaiah` (or your Hugging Face username)
- **Password**: **Paste your Access Token** (NOT your password!)

### Alternative: Use Token in URL

You can also embed the token in the URL:

```powershell
# Replace YOUR_TOKEN with your actual token
git remote set-url origin https://YOUR_TOKEN@huggingface.co/spaces/Velugondaiah/FaceDetection

# Then push
git push origin main
```

## Verify Deployment

After pushing:

1. Go to: https://huggingface.co/spaces/Velugondaiah/FaceDetection
2. Click "Logs" tab to see build progress
3. Wait 15-30 minutes for first build
4. Check "App" tab when it's running

## Your API will be at:
https://velugondaiah-facedetection.hf.space

