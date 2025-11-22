# Deploy to Hugging Face Spaces - Step by Step Guide

## Prerequisites Checklist

- [x] Dockerfile is ready
- [x] requirements.txt is complete
- [x] index.py is configured
- [x] README.md has proper metadata
- [ ] Git repository initialized
- [ ] Hugging Face account created
- [ ] Hugging Face Access Token created

## Step 1: Create Hugging Face Access Token

1. Go to https://huggingface.co/settings/tokens
2. Click "New token"
3. Name it (e.g., "face-detection-deploy")
4. Select "Write" permissions
5. Copy the token (you'll need it for git push)

## Step 2: Create a Hugging Face Space

1. Go to https://huggingface.co/spaces
2. Click "Create new Space"
3. Fill in:
   - **Space name**: `face-detection-api` (or your choice)
   - **SDK**: Select **Docker**
   - **Hardware**: 
     - For testing: **CPU Basic** (free)
     - For production: **CPU Upgrade** or **GPU** (paid)
   - **Visibility**: Public or Private
4. Click "Create Space"

## Step 3: Initialize Git and Push

### Option A: If repository is NOT initialized

```powershell
cd C:\Users\Kushvanth\OneDrive\Desktop\Detection\detection

# Initialize git (if not already done)
git init

# Add all files
git add .

# Commit
git commit -m "Initial commit: Face Detection API for Hugging Face"

# Add Hugging Face remote (replace YOUR_USERNAME and YOUR_SPACE_NAME)
git remote add origin https://huggingface.co/spaces/YOUR_USERNAME/YOUR_SPACE_NAME

# Push to Hugging Face
git push origin main
```

### Option B: If repository is already initialized

```powershell
cd C:\Users\Kushvanth\OneDrive\Desktop\Detection\detection

# Check current remotes
git remote -v

# If origin exists, remove it first
git remote remove origin

# Add Hugging Face remote (replace YOUR_USERNAME and YOUR_SPACE_NAME)
git remote add origin https://huggingface.co/spaces/YOUR_USERNAME/YOUR_SPACE_NAME

# Push to Hugging Face
git push origin main
```

**When prompted for credentials:**
- Username: Your Hugging Face username
- Password: Your Hugging Face Access Token (NOT your password)

## Step 4: Configure Environment Variables (Recommended)

1. Go to your Space page: `https://huggingface.co/spaces/YOUR_USERNAME/YOUR_SPACE_NAME`
2. Click "Settings" tab
3. Scroll to "Variables and secrets"
4. Add these secrets:
   - `SUPABASE_URL`: `https://nrxqcfdbyscqgrdrqegu.supabase.co`
   - `SUPABASE_KEY`: `eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...` (your full key)

**Note:** Your `index.py` already has defaults, but using secrets is more secure.

## Step 5: Monitor the Build

1. After pushing, go to your Space page
2. Click the "Logs" tab
3. Watch the build progress
4. **First build takes 15-30 minutes** (compiling dlib)
5. You'll see:
   - Docker build progress
   - Package installations
   - Application startup

## Step 6: Test Your Deployed API

Once the build completes and shows "Running", test:

1. **Health Check:**
   ```
   https://YOUR_USERNAME-YOUR_SPACE_NAME.hf.space/health
   ```

2. **API Info:**
   ```
   https://YOUR_USERNAME-YOUR_SPACE_NAME.hf.space/
   ```

3. **Interactive Docs:**
   ```
   https://YOUR_USERNAME-YOUR_SPACE_NAME.hf.space/docs
   ```

## Step 7: Update Your Express.js Backend

Update your Express.js code to use the Hugging Face URL:

```javascript
// For Hugging Face deployment
const FACE_DETECTION_URL = process.env.FACE_DETECTION_URL || 'https://YOUR_USERNAME-YOUR_SPACE_NAME.hf.space/detect-from-url/';
```

## Troubleshooting

### Build Fails
- Check logs for specific errors
- Ensure all files are committed
- Verify Dockerfile syntax

### API Not Starting
- Check logs for runtime errors
- Verify environment variables are set
- Ensure port is 7860

### Slow Build
- First build is slow (compiling dlib)
- Subsequent builds are faster (cached layers)

## Updating Your Deployment

After making changes:

```powershell
git add .
git commit -m "Update: description of changes"
git push origin main
```

Hugging Face will automatically rebuild.

## Important Notes

1. **Build Time**: First build takes 15-30 minutes due to dlib compilation
2. **Hardware**: CPU Basic is free but slower. Consider CPU Upgrade for production
3. **Secrets**: Use environment variables for sensitive data
4. **Logs**: Always check logs if something doesn't work
5. **Restart**: You can restart the Space from the Settings tab

