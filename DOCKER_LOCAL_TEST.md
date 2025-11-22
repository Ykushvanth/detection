# Local Docker Testing Guide

This guide will help you test your Docker container locally using Docker Desktop before deploying to Hugging Face.

## Prerequisites

- Docker Desktop installed and running
- Your project files ready

## Quick Start

### Option 1: Using PowerShell Script (Recommended for Windows)

1. Open PowerShell in the `detection` folder
2. Run:
   ```powershell
   .\docker-run-local.ps1
   ```

### Option 2: Using Batch Script

1. Double-click `docker-run-local.bat` or run it from Command Prompt

### Option 3: Manual Commands

1. **Build the Docker image:**
   ```bash
   docker build -t face-detection-api .
   ```

2. **Run the container:**
   ```bash
   docker run -p 7860:7860 face-detection-api
   ```

   Or with environment variables:
   ```bash
   docker run -p 7860:7860 -e SUPABASE_URL=your_url -e SUPABASE_KEY=your_key face-detection-api
   ```

## Testing the API

Once the container is running, you can test the API:

### 1. Health Check
Open your browser and go to:
```
http://localhost:7860/health
```

Or use curl:
```bash
curl http://localhost:7860/health
```

Expected response:
```json
{"status": "healthy"}
```

### 2. Root Endpoint
```
http://localhost:7860/
```

### 3. API Documentation (Interactive)
```
http://localhost:7860/docs
```

This will open the FastAPI interactive documentation where you can test all endpoints.

### 4. Test Upload Reference Image

Using curl:
```bash
curl -X POST "http://localhost:7860/upload-reference/" -F "file=@path/to/your/image.jpg"
```

### 5. Test Detect from URL

Using curl:
```bash
curl -X POST "http://localhost:7860/detect-from-url/" -H "Content-Type: application/json" -d "{\"url\": \"https://example.com/image.jpg\"}"
```

## Troubleshooting

### Docker Build Fails

1. **Check Docker Desktop is running:**
   - Open Docker Desktop
   - Ensure it shows "Docker Desktop is running"

2. **Check for errors in the build output:**
   - Look for specific error messages
   - Common issues:
     - Network connectivity (downloading dependencies)
     - Insufficient disk space
     - Missing files (Dockerfile, requirements.txt, etc.)

3. **If build fails with CMake error:**
   - The Dockerfile includes all necessary dependencies
   - Try rebuilding: `docker build --no-cache -t face-detection-api .`

### Container Won't Start

1. **Check if port 7860 is already in use:**
   ```powershell
   netstat -ano | findstr :7860
   ```
   If something is using the port, either:
   - Stop the other service
   - Change the port mapping: `docker run -p 8080:7860 face-detection-api` (then access at http://localhost:8080)

2. **Check Docker logs:**
   ```bash
   docker ps -a
   docker logs <container_id>
   ```

### API Not Responding

1. **Verify the container is running:**
   ```bash
   docker ps
   ```

2. **Check container logs:**
   ```bash
   docker logs <container_id>
   ```

3. **Test from inside the container:**
   ```bash
   docker exec -it <container_id> /bin/bash
   curl http://localhost:7860/health
   ```

### Environment Variables

If you need to override Supabase credentials:
```bash
docker run -p 7860:7860 -e SUPABASE_URL=your_url -e SUPABASE_KEY=your_key face-detection-api
```

## Stopping the Container

- Press `Ctrl+C` in the terminal where the container is running
- Or in another terminal:
  ```bash
  docker ps
  docker stop <container_id>
  ```

## Cleaning Up

After testing, you can clean up:

```bash
# Stop and remove containers
docker ps -a
docker rm <container_id>

# Remove the image (optional)
docker rmi face-detection-api
```

## Notes

- The first build may take 15-20 minutes as it downloads all dependencies and compiles dlib
- Subsequent builds will be faster due to Docker layer caching
- The container uses port 7860 (Hugging Face Spaces default)
- Reference images are stored in the `reference_images` directory inside the container
- Default Supabase credentials are included for local testing (replace with environment variables for production)

## Next Steps

Once local testing is successful:

1. Verify all endpoints work correctly
2. Test with actual images
3. Check the logs for any errors
4. Follow the `DEPLOYMENT.md` guide to deploy to Hugging Face


