# Deployment Guide for Hugging Face Spaces

This guide will help you deploy your Face Detection API to Hugging Face Spaces using Docker.

## Prerequisites

1. A Hugging Face account (sign up at https://huggingface.co/)
2. Docker installed locally (optional, for testing)
3. Git installed

## Step 1: Prepare Your Repository

Ensure your repository contains:
- `Dockerfile` ✅
- `requirements.txt` ✅
- `index.py` ✅
- `README.md` ✅
- `.dockerignore` ✅

## Step 2: Create a Hugging Face Space

1. Go to https://huggingface.co/spaces
2. Click "Create new Space"
3. Fill in the details:
   - **Space name**: `face-detection-api` (or your preferred name)
   - **SDK**: Select **Docker**
   - **Hardware**: Choose based on your needs (CPU Basic, CPU Upgrade, or GPU if needed)
   - **Visibility**: Public or Private
4. Click "Create Space"

## Step 3: Push Your Code to Hugging Face

### Option A: Using Git (Recommended)

1. **Initialize Git repository** (if not already initialized):
```bash
cd detection
git init
git add .
git commit -m "Initial commit for Hugging Face deployment"
```

2. **Add Hugging Face remote**:
```bash
git remote add origin https://huggingface.co/spaces/YOUR_USERNAME/YOUR_SPACE_NAME
```

Replace `YOUR_USERNAME` with your Hugging Face username and `YOUR_SPACE_NAME` with your Space name.

3. **Push to Hugging Face**:
```bash
git push origin main
```

If prompted for authentication:
- Use your Hugging Face username
- Use a Hugging Face Access Token (create one at https://huggingface.co/settings/tokens)

### Option B: Using Hugging Face Web Interface

1. Go to your Space page on Hugging Face
2. Click "Files and versions" tab
3. Upload files directly using the web interface
4. Upload files in this order:
   - `Dockerfile`
   - `requirements.txt`
   - `index.py`
   - `README.md`
   - `.dockerignore`

## Step 4: Configure Environment Variables (if needed)

If you want to use environment variables instead of hardcoded values:

1. Go to your Space settings
2. Navigate to "Variables and secrets"
3. Add your environment variables:
   - `SUPABASE_URL`: Your Supabase URL
   - `SUPABASE_KEY`: Your Supabase key
   - `PORT`: 7860 (optional, this is the default)

Then update `index.py` to read from environment variables:
```python
import os
supabase_url = os.getenv("SUPABASE_URL", "https://nrxqcfdbyscqgrdrqegu.supabase.co")
supabase_key = os.getenv("SUPABASE_KEY", "your-default-key")
```

## Step 5: Monitor Build and Deployment

1. After pushing, go to your Space page
2. Click on the "Logs" tab to see the build process
3. Wait for the build to complete (this may take 5-10 minutes the first time)
4. Once built, your API will be available at:
   ```
   https://YOUR_USERNAME-YOUR_SPACE_NAME.hf.space
   ```

## Step 6: Test Your Deployment

Once deployed, test your API:

1. **Health Check**:
   ```
   GET https://YOUR_USERNAME-YOUR_SPACE_NAME.hf.space/health
   ```

2. **Root Endpoint**:
   ```
   GET https://YOUR_USERNAME-YOUR_SPACE_NAME.hf.space/
   ```

3. **API Documentation** (FastAPI auto-generated):
   ```
   https://YOUR_USERNAME-YOUR_SPACE_NAME.hf.space/docs
   ```

## Troubleshooting

### Build Fails

1. Check the logs in the "Logs" tab
2. Common issues:
   - Missing dependencies in `requirements.txt`
   - Dockerfile syntax errors
   - Port conflicts (ensure port is 7860)

### API Not Responding

1. Check if the build completed successfully
2. Verify the port is set to 7860 in the Dockerfile
3. Check the logs for runtime errors

### Slow Build Times

- First build takes longer due to downloading base images and dependencies
- Subsequent builds are faster due to caching
- Consider using a smaller base image or optimizing your Dockerfile

## Local Testing Before Deployment

Before deploying, you can test the Docker image locally:

```bash
# Build the image
docker build -t face-detection-api .

# Run the container
docker run -p 7860:7860 face-detection-api

# Test the API
curl http://localhost:7860/health
```

## Updating Your Deployment

To update your deployment:

```bash
git add .
git commit -m "Update: description of changes"
git push origin main
```

Hugging Face will automatically rebuild and redeploy your Space.

## Additional Resources

- [Hugging Face Spaces Documentation](https://huggingface.co/docs/hub/spaces)
- [Docker Documentation](https://docs.docker.com/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)



